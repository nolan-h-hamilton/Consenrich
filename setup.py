import sys
import os
import glob
import shutil
import subprocess
import sysconfig
import textwrap
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
import numpy


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDORED_HTSLIB_DIR = os.path.join(ROOT_DIR, "vendor", "htslib")
PREFIX_INCLUDE_DIR = os.path.join(sys.prefix, "include")
PREFIX_LIB_DIR = os.path.join(sys.prefix, "lib")


def get_includes():
    return [
        numpy.get_include(),
        os.path.join("src", "consenrich"),
    ] + getHtslibIncludeDirs()


def hasVendoredHtslib(htslibDir=VENDORED_HTSLIB_DIR):
    return os.path.exists(os.path.join(htslibDir, "Makefile"))


def getHtslibIncludeDirs(htslibDir=VENDORED_HTSLIB_DIR):
    includeDirs = []
    if hasVendoredHtslib(htslibDir):
        includeDirs.extend(
            [
                htslibDir,
                os.path.join(htslibDir, "htslib"),
            ]
        )
    includeDirs.extend([PREFIX_INCLUDE_DIR, os.path.join(PREFIX_INCLUDE_DIR, "htslib")])
    return includeDirs


def get_library_dirs():
    return [PREFIX_LIB_DIR]


def getBundledHtslibArchive(htslibDir=VENDORED_HTSLIB_DIR):
    return os.path.join(htslibDir, "libhts.a")


def findStaticLibrary(libraryName):
    candidateDirs = [
        PREFIX_LIB_DIR,
        "/usr/local/lib",
        "/opt/homebrew/lib",
        "/usr/lib",
        "/usr/lib64",
        "/lib",
        "/lib64",
    ]
    libraryPattern = f"lib{libraryName}.a"
    for candidateDir in candidateDirs:
        candidatePath = os.path.join(candidateDir, libraryPattern)
        if os.path.exists(candidatePath):
            return candidatePath
    globPatterns = [
        f"/usr/lib/*/{libraryPattern}",
        f"/lib/*/{libraryPattern}",
        f"/usr/local/lib/*/{libraryPattern}",
    ]
    for pattern in globPatterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def getBundledDependencyArchives():
    dependencyArchives = []
    if sys.platform == "darwin":
        staticZlib = findStaticLibrary("z")
        if staticZlib is not None:
            dependencyArchives.append(staticZlib)
    return dependencyArchives


def getBundledHtslibLibraries():
    libraries = []
    if sys.platform != "darwin" or findStaticLibrary("z") is None:
        libraries.append("z")
    if sys.platform.startswith("linux"):
        libraries.extend(["m", "pthread"])
    return libraries


def getBundledHtslibExtraObjects():
    return [getBundledHtslibArchive()] + getBundledDependencyArchives()


def writeTextIfChanged(path, contents):
    existingContents = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            existingContents = handle.read()
    if existingContents == contents:
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(contents)


def joinCompilerFlags(*flagGroups):
    return " ".join(flag for flagGroup in flagGroups for flag in flagGroup if flag)


def parseBoolEnv(name, defaultValue=False):
    rawValue = os.environ.get(name)
    if rawValue is None:
        return defaultValue
    normalized = rawValue.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be one of 1, true, yes, on, 0, false, no, off")


def parsePositiveIntEnv(name, defaultValue):
    rawValue = os.environ.get(name)
    if rawValue is None or rawValue.strip() == "":
        return defaultValue
    parsedValue = int(rawValue)
    if parsedValue < 1:
        raise ValueError(f"{name} must be positive")
    return parsedValue


def findDarwinLibompPrefix():
    envPrefix = os.environ.get("CONSENRICH_LIBOMP_PREFIX")
    candidatePrefixes = [envPrefix] if envPrefix else [
        "/opt/homebrew/opt/libomp",
        "/usr/local/opt/libomp",
    ]
    for candidatePrefix in candidatePrefixes:
        includePath = os.path.join(candidatePrefix, "include", "omp.h")
        libraryPath = os.path.join(candidatePrefix, "lib", "libomp.dylib")
        if os.path.exists(includePath) and os.path.exists(libraryPath):
            return candidatePrefix
    raise RuntimeError(
        "CONSENRICH_USE_OPENMP=1 on macOS requires libomp. "
        "Install libomp or set CONSENRICH_LIBOMP_PREFIX."
    )


def getOpenMPConfig(useOpenMP):
    if not useOpenMP:
        return [], [], [], []
    if sys.platform == "darwin":
        libompPrefix = findDarwinLibompPrefix()
        libompLibDir = os.path.join(libompPrefix, "lib")
        return (
            ["-Xpreprocessor", "-fopenmp"],
            ["-lomp", f"-Wl,-rpath,{libompLibDir}"],
            [os.path.join(libompPrefix, "include")],
            [libompLibDir],
        )
    if sys.platform.startswith("linux"):
        return ["-fopenmp"], ["-fopenmp"], [], []
    raise RuntimeError(f"CONSENRICH_USE_OPENMP=1 is not configured for {sys.platform}")


def getVendoredHtslibCPPFlags():
    return joinCompilerFlags(
        [f"-I{PREFIX_INCLUDE_DIR}"],
        [os.environ.get("CPPFLAGS", "").strip()],
    )


def getVendoredHtslibCFlags():
    baseCFlags = [
        "-g",
        "-Wall",
        "-O2",
        "-fvisibility=hidden",
        "-fPIC",
    ]
    if sys.platform == "darwin":
        deploymentTarget = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if not deploymentTarget:
            deploymentTarget = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
        if deploymentTarget:
            baseCFlags.append(f"-mmacosx-version-min={deploymentTarget}")
    extraCFlags = os.environ.get("CFLAGS", "").strip()
    return joinCompilerFlags(baseCFlags, [extraCFlags])


def getVendoredHtslibLdFlags():
    return joinCompilerFlags([os.environ.get("LDFLAGS", "").strip()])


def getVendoredHtslibConfigMk():
    return textwrap.dedent(
        f"""\
        CC = {os.environ.get("CC", "cc")}
        RANLIB = {os.environ.get("RANLIB", "ranlib")}

        CPPFLAGS = {getVendoredHtslibCPPFlags()}
        CFLAGS = {getVendoredHtslibCFlags()}
        LDFLAGS = {getVendoredHtslibLdFlags()}
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
        """
    )


def getVendoredHtslibConfigH():
    return textwrap.dedent(
        """\
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
        """
    )


def prepareVendoredHtslibBuild(htslibDir):
    writeTextIfChanged(os.path.join(htslibDir, "config.mk"), getVendoredHtslibConfigMk())
    writeTextIfChanged(os.path.join(htslibDir, "config.h"), getVendoredHtslibConfigH())


def buildVendoredHtslib(htslibDir):
    if not hasVendoredHtslib(htslibDir):
        raise FileNotFoundError("Vendored htslib source tree is missing")
    prepareVendoredHtslibBuild(htslibDir)
    subprocess.check_call(
        ["make", "-C", htslibDir, "clean"],
        cwd=ROOT_DIR,
    )
    prepareVendoredHtslibBuild(htslibDir)
    subprocess.check_call(
        ["make", "-C", htslibDir, "lib-static"],
        cwd=ROOT_DIR,
    )
    if not os.path.exists(getBundledHtslibArchive(htslibDir)):
        raise FileNotFoundError("Failed to build vendored libhts.a")


base_compile = [
    "-O3",
    "-fno-trapping-math",
    "-fno-math-errno",
    "-mtune=generic",
]

useOpenMP = parseBoolEnv("CONSENRICH_USE_OPENMP", False)
openMPFactorMinRows = parsePositiveIntEnv(
    "CONSENRICH_OPENMP_FACTOR_MIN_ROWS", 262_144
)
openMPApplyMinRows = parsePositiveIntEnv(
    "CONSENRICH_OPENMP_APPLY_MIN_ROWS", 1_048_576
)
openMPCompileArgs, openMPLinkArgs, openMPIncludeDirs, openMPLibraryDirs = (
    getOpenMPConfig(useOpenMP)
)


class buildConsenrichExt(build_ext):
    def build_extensions(self):
        if hasVendoredHtslib():
            htslibBuildDir = self.copyVendoredHtslib()
            buildVendoredHtslib(htslibBuildDir)
            for extension in self.extensions:
                extension.include_dirs = self.replaceVendoredHtslibPaths(
                    extension.include_dirs,
                    htslibBuildDir,
                )
                extension.extra_objects = self.replaceVendoredHtslibPaths(
                    extension.extra_objects,
                    htslibBuildDir,
                )
        super().build_extensions()

    def copyVendoredHtslib(self):
        htslibBuildDir = os.path.join(self.build_temp, "vendor", "htslib")
        if os.path.isdir(htslibBuildDir):
            shutil.rmtree(htslibBuildDir)
        os.makedirs(os.path.dirname(htslibBuildDir), exist_ok=True)
        shutil.copytree(
            VENDORED_HTSLIB_DIR,
            htslibBuildDir,
            ignore=shutil.ignore_patterns(".git", "*.o", "*.a"),
        )
        return htslibBuildDir

    def replaceVendoredHtslibPaths(self, paths, htslibBuildDir):
        return [
            path.replace(VENDORED_HTSLIB_DIR, htslibBuildDir, 1)
            if isinstance(path, str) and path.startswith(VENDORED_HTSLIB_DIR)
            else path
            for path in paths
        ]


class buildConsenrichPy(build_py):
    def run(self):
        packageBuildDir = os.path.join(self.build_lib, "consenrich")
        if os.path.isdir(packageBuildDir):
            shutil.rmtree(packageBuildDir)
        super().run()


extensions = [
    Extension(
        "consenrich.cconsenrich",
        sources=["src/consenrich/cconsenrich.pyx"],
        include_dirs=get_includes() + openMPIncludeDirs,
        libraries=getBundledHtslibLibraries(),
        library_dirs=get_library_dirs() + openMPLibraryDirs,
        extra_objects=getBundledHtslibExtraObjects(),
        extra_compile_args=base_compile + openMPCompileArgs,
        extra_link_args=openMPLinkArgs,
    ),
    Extension(
        "consenrich.ccounts",
        sources=[
            "src/consenrich/ccounts.pyx",
            "src/consenrich/native/ccounts_backend.c",
        ],
        include_dirs=[numpy.get_include(), os.path.join("src", "consenrich")]
        + getHtslibIncludeDirs(),
        libraries=getBundledHtslibLibraries(),
        library_dirs=get_library_dirs(),
        extra_objects=getBundledHtslibExtraObjects(),
        extra_compile_args=base_compile,
    ),
    Extension(
        "consenrich.cuncertainty",
        sources=["src/consenrich/cuncertainty.pyx"],
        include_dirs=[
            numpy.get_include(),
            os.path.join("src", "consenrich"),
        ] + openMPIncludeDirs,
        library_dirs=openMPLibraryDirs,
        extra_compile_args=base_compile + openMPCompileArgs,
        extra_link_args=openMPLinkArgs,
    ),
]

extModules = cythonize(
    extensions,
    build_dir=os.path.join("build", "cython"),
    language_level="3",
    compile_time_env={
        "USE_OPENMP": useOpenMP,
        "OPENMP_FACTOR_MIN_ROWS": openMPFactorMinRows,
        "OPENMP_APPLY_MIN_ROWS": openMPApplyMinRows,
    },
)
for extension in extModules:
    extension.depends = [
        dependency for dependency in extension.depends if not os.path.isabs(dependency)
    ]
for generatedPath in [
    "src/consenrich/cconsenrich.c",
    "src/consenrich/ccounts.c",
    "src/consenrich/cuncertainty.c",
]:
    if os.path.exists(generatedPath):
        os.remove(generatedPath)


setup(
    ext_modules=extModules,
    cmdclass={
        "build_ext": buildConsenrichExt,
        "build_py": buildConsenrichPy,
    },
    zip_safe=False,
)
