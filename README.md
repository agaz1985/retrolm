<p align="center">
  <img src="imgs/retrolm_logo.png" alt="retrolm Logo" width="600"/>
</p>

# 💻 retrolm: The PII Processor Project

### **[ STATUS: BARE METAL ALPHA ]**
---

## 💾 Modern AI on Ancient Silicon. *Phat.*

**retrolm** is a hardcore, "bare metal" Large Language Model inference engine built entirely from scratch in **C and hand-optimized x86 Assembly**. This ain't your grandma's Python script.

Designed to breathe life into the majestic **Intel Pentium II** architecture and compiled exclusively using the legendary **Turbo C (Borland v4.5+)**, this project attempts the impossible: running a foundational transformer model on a machine whose primary purpose was running *Duke Nukem 3D* and accessing AOL.

*There are no JITs. There is no CUDA. There are only registers, manual stack management, and a lot of caffeine.*

## 💀 Development Philosophy: The Hard Way (No Cheats!)

We're going full retro development. In the true spirit of optimization and pain, this codebase was forged with **zero modern safety nets**:

* **🚫 LLM Free Zone:** Every single line of `retrolm` code was written by a human. No ChatGPT, no GitHub Copilot. We built an LLM without an LLM. **Boom!**
* **💾 Text Editor Only:** Compiled from code written in a plain text editor. No fancy IDE, no code completion, no syntax highlighting. If it compiled, it shipped.
* **📚 Analog Resources:** The only documentation allowed were printed manuals: K&R C, the Intel PII Developer's Guides, and the deepest archives of Usenet and Stack Overflow.

---

## 🖥️ Required Gear (To Run the Metal)

To compile and run `retrolm` and truly experience the pain, you'll need the following vintage specifications:

| Component | Minimum Spec | Recommended Spec |
| :--- | :--- | :--- |
| **Processor** | Intel Pentium II (233MHz) | Intel Pentium II Deschutes (450MHz) |
| **RAM** | 32 MB EDO or SDRAM | 128 MB PC100 SDRAM |
| **OS** | MS-DOS 6.22 or Win95/98 | Windows 98 SE |
| **Storage** | 10 MB Free HDD Space | 50 MB Free HDD Space |
| **Compiler** | Turbo C/C++ 4.5+ | Borland C++ 5.02 (The Big One) |
| **Transfer** | 3.5" Floppy Disk | 100 MB Iomega Zip Drive |

---

### **WHY?**
Because we spent so much time asking if we *could* run LLMs on 4090s, we never stopped to ask if we **should** run them on a beige tower with 64MB of EDO RAM.