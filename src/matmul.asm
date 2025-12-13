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
    
    mov eax, [ebp+8]   ; First argument
    add eax, [ebp+12]  ; Second argument
    
    pop ebp
    ret