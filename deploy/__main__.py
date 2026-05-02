"""Pulumi program to deploy agenttester remote-execution infrastructure on AWS."""

from pathlib import Path

import pulumi
import pulumi_aws as aws

# ── config ────────────────────────────────────────────────────────────

config = pulumi.Config()
instance_type = config.get("instance_type") or "t3.large"
instance_count = config.get_int("instance_count") or 1
ssh_pub_key_path = config.get("ssh_public_key_path") or "~/.ssh/id_ed25519.pub"
allowed_ssh_cidrs = config.require_object("allowed_ssh_cidrs")

ssh_pub_key = Path(ssh_pub_key_path).expanduser().read_text().strip()

# ── AMI (latest Ubuntu 22.04) ─────────────────────────────────────────

ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical
    filters=[
        aws.ec2.GetAmiFilterArgs(
            name="name",
            values=["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"],
        ),
        aws.ec2.GetAmiFilterArgs(
            name="virtualization-type",
            values=["hvm"],
        ),
    ],
)

# ── key pair ──────────────────────────────────────────────────────────

key_pair = aws.ec2.KeyPair(
    "agenttester-key",
    public_key=ssh_pub_key,
)

# ── security group (SSH only) ─────────────────────────────────────────

sg = aws.ec2.SecurityGroup(
    "agenttester-sg",
    description="Allow SSH for agenttester remote agents",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=22,
            to_port=22,
            cidr_blocks=allowed_ssh_cidrs,
            description="SSH",
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1",
            from_port=0,
            to_port=0,
            cidr_blocks=["0.0.0.0/0"],
            description="All outbound",
        ),
    ],
)

# ── IAM role + instance profile ───────────────────────────────────────

assume_role_policy = """{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}"""

role = aws.iam.Role(
    "agenttester-role",
    assume_role_policy=assume_role_policy,
    tags={"Project": "agenttester"},
)

# SSM managed policy so you can debug via Session Manager if needed
aws.iam.RolePolicyAttachment(
    "agenttester-ssm",
    role=role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
)

instance_profile = aws.iam.InstanceProfile(
    "agenttester-profile",
    role=role.name,
)

# ── user data ─────────────────────────────────────────────────────────

user_data_script = (Path(__file__).parent / "user_data.sh").read_text()

# ── EC2 instances ─────────────────────────────────────────────────────

instances = []
for i in range(instance_count):
    inst = aws.ec2.Instance(
        f"agenttester-agent-{i}",
        ami=ami.id,
        instance_type=instance_type,
        key_name=key_pair.key_name,
        vpc_security_group_ids=[sg.id],
        iam_instance_profile=instance_profile.name,
        user_data=user_data_script,
        root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
            volume_size=50,
            volume_type="gp3",
        ),
        tags={
            "Name": f"agenttester-agent-{i}",
            "Project": "agenttester",
        },
    )
    instances.append(inst)

# ── outputs ───────────────────────────────────────────────────────────

pulumi.export("instance_ids", [inst.id for inst in instances])
pulumi.export("public_ips", [inst.public_ip for inst in instances])
pulumi.export(
    "ssh_hosts",
    [inst.public_ip.apply(lambda ip: f"ubuntu@{ip}") for inst in instances],
)
pulumi.export(
    "agenttester_yaml_snippet",
    pulumi.Output.all(*[inst.public_ip for inst in instances]).apply(
        lambda ips: "\n".join(
            f"  remote-agent-{i}:\n"
            f'    command: \'claude -p {{prompt}} --allowedTools "Bash,Read,Edit"\'\n'
            f"    host: ubuntu@{ip}\n"
            f"    remote_workdir: /tmp/agenttester\n"
            f"    commit_style: auto\n"
            f"    timeout: 600"
            for i, ip in enumerate(ips)
        )
    ),
)
