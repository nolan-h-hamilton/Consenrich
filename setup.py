import os
import shutil
import sys
from setuptools import Extension, setup
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
import numpy

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


class buildConsenrichPy(build_py):
    def run(self):
        packageBuildDir = os.path.join(self.build_lib, "consenrich")
        if os.path.isdir(packageBuildDir):
            shutil.rmtree(packageBuildDir)
        super().run()


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


extensions = [
    Extension(
        f"consenrich.{module}",
        sources=[f"src/consenrich/{module}.pyx"],
        include_dirs=[numpy.get_include(), "src/consenrich"] + openMPIncludeDirs,
        library_dirs=openMPLibraryDirs,
        extra_compile_args=base_compile + openMPCompileArgs,
        extra_link_args=openMPLinkArgs,
    )
    for module in ("cconsenrich", "cuncertainty")
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
    extension.depends = [value for value in extension.depends if not os.path.isabs(value)]
setup(ext_modules=extModules, cmdclass={"build_py": buildConsenrichPy}, zip_safe=False)
