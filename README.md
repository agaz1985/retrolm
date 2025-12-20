<p align="center">
  <img src="imgs/retrolm_logo.png" alt="retrolm Logo" width="600"/>
</p>

# 🧠 Bringing Modern AI Inference to Ancient Silicon

**retrolm** is a transformer-based LLM inference engine written from scratch in **C and x86 Assembly**, designed to run on **Intel Pentium II** hardware with **FreeDOS**.

## 🛠️ The Challenge

* **Zero Dependencies:** No PyTorch, NumPy, or BLAS—all linear algebra written from scratch
* **No AI Assistance:** Core implementation written without LLMs *(documentation and tests were LLM-assisted for efficiency)*
* **32-bit x86 Assembly:** Hand-optimized routines for critical operations
* **Classic References:** K&R's *C Programming Language* and Hyde's *Art of Assembly Language*
* **Cross-Compiled:** DJGPP toolchain targeting FreeDOS

---

## 🖥️ Target Hardware

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **CPU** | Pentium II 233MHz | Pentium II 400MHz+ |
| **RAM** | 64 MB | 128 MB |
| **OS** | FreeDOS 1.2+ | FreeDOS 1.3 |

---

## 🔧 Quick Start

**Prerequisites:** Docker Desktop

```bash
# Clone and setup
git clone https://github.com/agaz1985/retrolm.git
cd retrolm
chmod +x *.sh
make build

# Development
make run          # Fast: Build & run on Linux (32-bit)
make dos          # Deploy: Build retrolm.exe for FreeDOS
make shell        # Debug: Interactive container shell
make clean        # Remove build artifacts
```

**Deploy to Pentium II:**
1. `make dos` → creates `build/retrolm.exe`
2. Transfer via USB/floppy/serial to FreeDOS machine
3. Run: `C:\> retrolm.exe`

**Toolchain:** NASM (asm) • GCC `-m32` (Linux) • DJGPP (DOS) • Docker/Ubuntu 22.04

---

## 🤔 Why?

An exploration of what happens when modern AI meets 1998 hardware constraints—deep diving into low-level optimization, memory management, and vintage architecture.

---

## 📚 References

- *The C Programming Language* (2nd Ed.) - Kernighan & Ritchie
- *The Art of Assembly Language* - Randall Hyde

Built with determination, coffee, and deep appreciation for vintage computing. ☕