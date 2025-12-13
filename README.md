<p align="center">
  <img src="imgs/retrolm_logo.png" alt="retrolm Logo" width="600"/>
</p>

# 🧠 Bringing Modern AI Inference to Ancient Silicon

**retrolm** is a personal, challenging side-project: a complete Large Language Model (LLM) inference engine written from the ground up in **C and x86 Assembly**.

The core goal is to run a simple transformer model on a machine it was never meant to touch: the **Intel Pentium II** architecture, compiled using the classic **Turbo C** toolchain. This project explores the true engineering overhead of deep learning by stripping away all modern frameworks and dependencies.

## 🛠️ The Development Ethos (Hard Mode Activated)

In building this, I deliberately chose constraints to understand the fundamentals and maximize the optimization challenge:

* **Solo Development:** Every line of code was written by me alone.
* **No AI Assistance:** I relied purely on my own knowledge and classic documentation, avoiding LLMs for coding the LLM.
* **Minimalist Tooling:** Development was done primarily using a plain text editor, bypassing the comfort of modern IDE features like advanced autocompletion or dynamic debugging.
* **Classic References:** My primary guides were physical manuals: *The C Programming Language* (K&R), the *The Art of Assembly Language* (H), and archived technical forums.

## 💾 Core Technical Details

* **Zero Dependencies:** The project contains no external libraries (no PyTorch, NumPy, or specialized BLAS). I wrote all custom math and linear algebra kernels myself.
* **Authentic Toolchain:** The code is structured and compiled to be fully compatible with the Borland Turbo C / C++ environment, targeting DOS/Win9x systems.

---

# 🖥️ Required Target Hardware

To run and compile `retrolm`, you'll be aiming for the following vintage specifications. Results will vary *greatly* based on clock speed and available RAM.

| Component | Minimum Spec | Recommended Spec |
| :--- | :--- | :--- |
| **Processor** | Intel Pentium II (233MHz) | Intel Pentium II Deschutes (450MHz) |
| **RAM** | 32 MB EDO or SDRAM | 128 MB PC100 SDRAM |
| **OS** | MS-DOS 6.22 or Win95/98 | Windows 98 SE |
| **Compiler** | Turbo C/C++ 4.5+ | Borland C++ 5.02 |

---

## 🔧 Development Environment

While the target is vintage hardware, development happens on modern systems using a Docker-based cross-compilation toolchain.

### Prerequisites

- **Docker Desktop** (for Mac, Windows, or Linux)
- A text editor of your choice
- Git (optional, for version control)

### Project Structure
```
retrolm/
├── Dockerfile              # Docker build configuration
├── Makefile               # Build automation
├── build-linux.sh         # Linux build script (for testing)
├── build-dos.sh           # DOS build script (for deployment)
├── docker-shell.sh        # Interactive container access
├── src/
│   ├── matmul.asm        # x86 assembly (32-bit)
│   ├── exceptions.c
│   ├── logger.c
│   ├── main.c
│   ├── matrix.c
│   └── memory.c
└── build/                 # Generated executables (gitignored)
```

### Setup Instructions

1. **Clone the repository:**
```bash
   git clone <repository-url>
   cd retrolm
```

2. **Make scripts executable:**
```bash
   chmod +x build-linux.sh build-dos.sh docker-shell.sh
```

3. **Build the Docker image:**
```bash
   make build
```

### Development Workflow

**Fast iteration (development on Linux 32-bit):**
```bash
make run          # Build and run instantly for testing
```

**Build for FreeDOS deployment:**
```bash
make dos          # Creates build/retrollm.exe
```

**Interactive debugging:**
```bash
make shell        # Opens bash shell inside container
```

**Clean build artifacts:**
```bash
make clean        # Remove build directory
```

### Available Make Commands
```bash
make dev          # Build for Linux (32-bit testing)
make dos          # Build for FreeDOS (deployment)
make run          # Build and run Linux version
make shell        # Open interactive Docker shell
make clean        # Remove build artifacts
make clean-docker # Remove Docker image
make help         # Show all commands
```

### Assembly Code Architecture

The project uses conditional assembly to support both development (Linux ELF) and deployment (DOS COFF) formats:
```nasm
BITS 32

section .text

; Export with underscore for COFF (DOS), without for ELF (Linux)
%ifdef COFF
    global _add_numbers
    _add_numbers:
%else
    global add_numbers
    add_numbers:
%endif
    push ebp
    mov ebp, esp
    mov eax, [ebp+8]
    add eax, [ebp+12]
    pop ebp
    ret
```

### Build Toolchain Details

- **Assembler:** NASM (Netwide Assembler)
- **Linux Compiler:** GCC with `-m32` flag for 32-bit x86
- **DOS Cross-Compiler:** DJGPP (i586-pc-msdosdjgpp-gcc)
- **Docker Base:** Ubuntu 22.04 (linux/amd64)

### Development Platform

Development is performed on an **Apple M4 MacBook Air** using Docker for x86_64 emulation via Rosetta 2. The same Docker-based workflow works on any platform (Mac/Windows/Linux).

### Deployment to Target Hardware

1. Build the DOS executable:
```bash
   make dos
```

2. Transfer `build/retrollm.exe` to your Pentium II:
   - Copy to USB drive (if motherboard supports USB)
   - Copy to 3.5" floppy disk
   - Transfer via serial/parallel cable
   - Use network boot (if available)

3. Boot FreeDOS/DOS on the Pentium II and run:
```bash
   C:\> retrollm.exe
```

---

## 🤔 Why Build This?

I wanted to find out what happens when you take one of the most resource-intensive computing concepts of today and force it onto the constraints of 1998 hardware. It's an exploration of low-level optimization, memory management, and vintage computer architecture—a challenging exercise in doing things the hard way.

---

## 📚 References

- **The C Programming Language** (2nd Edition) - Kernighan & Ritchie
- **The Art of Assembly Language** - Randall Hyde
- Intel Pentium II Processor Specification
- DJGPP Cross-Compiler Documentation
- NASM Assembly Documentation

## 📝 License

[Your chosen license]

## 🙏 Acknowledgments

Built with determination, coffee, and a deep appreciation for the engineers who designed these incredible machines decades ago.