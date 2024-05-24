# Chatbot using RAG (Retrieval-Augmented Generation) in Django
## Assumptions
In this document, the following three conditions are assumed to assure operation.

1. The developers use the docker environment.
1. The host OS is Ubuntu 22.04 (64bit OS).

    ```bash
    cat /etc/lsb-release
    # DISTRIB_ID=Ubuntu
    # DISTRIB_RELEASE=22.04
    # DISTRIB_CODENAME=jammy
    # DISTRIB_DESCRIPTION="Ubuntu 22.04.4 LTS"
    ```

1. The host environment is Raspberry Pi 4 model B of arm64 (i.e. 64bit architecture).

    Note: Please enter the command `uname -a` on your terminal to check kernel version. The result is shown below:

    ```bash
    uname -a
    # Linux llm 5.15.0-1050-raspi #53-Ubuntu SMP PREEMPT Thu Mar 21 10:02:47 UTC 2024 aarch64 aarch64 aarch64 GNU/Linux
    ```

## Preparations
### Step1: Create `.env` files in the `env_files` directory
Please check the [Common README.md](./env_files/README.md), [Database README.md](./env_files/database/README.md), and [Backend README.md](./env_files/backend/README.md) for detail.

### Step2: Build images
Run the following command to create docker images.

```bash
# Current directory: rag-chatbot-django
./wrapper.sh build
```

### Step3: Setup backend
See the [README.md](./backend/README.md) in detail.

## Execution
Enter the following commands in your terminal to start Django application and Database server.

```bash
./wrapper.sh start
```

Then, access to `http://server-ip-address:7654` tio check the operation of Django application.
