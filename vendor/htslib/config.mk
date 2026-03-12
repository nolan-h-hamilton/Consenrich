CC = arm64-apple-darwin20.0.0-clang
RANLIB = arm64-apple-darwin20.0.0-ranlib

CPPFLAGS = -I/Users/nhh/miniforge3/envs/py312/include -D_FORTIFY_SOURCE=2 -isystem /Users/nhh/miniforge3/envs/py312/include
CFLAGS = -g -Wall -O2 -fvisibility=hidden -fPIC -mmacosx-version-min=11.0 -ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -isystem /Users/nhh/miniforge3/envs/py312/include
LDFLAGS = -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,/Users/nhh/miniforge3/envs/py312/lib -L/Users/nhh/miniforge3/envs/py312/lib
VERSION_SCRIPT_LDFLAGS =
LIBS = -lz -lm

NONCONFIGURE_OBJS =
plugin_OBJS =
noplugin_LDFLAGS =
noplugin_LIBS =

REF_CACHE_PROGRAMS =
HTS_CFLAGS_AVX2 =
HTS_CFLAGS_AVX512 =
HTS_CFLAGS_SSE4 =
