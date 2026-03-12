/* consenrich vendored htslib config */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#define HAVE_DRAND48 1
#if defined __x86_64__
#define HAVE_X86INTRIN_H 1
#endif
#if defined __x86_64__ || defined __arm__ || defined __aarch64__
#define HAVE_ATTRIBUTE_CONSTRUCTOR 1
#endif
#if defined __linux__
#define HAVE_GETAUXVAL 1
#elif defined __FreeBSD__
#define HAVE_ELF_AUX_INFO 1
#elif defined __OpenBSD__
#define HAVE_OPENBSD 1
#endif
