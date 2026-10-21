import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import struct
import urllib.request


def readRange(url, start, size):
    request = urllib.request.Request(
        f"{url}?range={start}-{start + size - 1}",
        headers={"Range": f"bytes={start}-{start + size - 1}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError(f"HTTP range response required: {url}")
        content = response.read(size + 1)
        if len(content) != size:
            raise ValueError(f"Incomplete range at byte {start}: {url}")
        return content, response.headers.get("Last-Modified")


def readAssembly(assembly, resourceDir):
    sizesPath = resourceDir / f"{assembly}.sizes"
    chromSizes = {}
    for line in sizesPath.read_text().splitlines():
        chrom, size = line.split()
        if chrom in chromSizes:
            raise ValueError(f"Duplicate chromosome: {sizesPath}: {chrom}")
        chromSizes[chrom] = int(size)
    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{assembly}/bigZips/{assembly}.2bit"
    header, lastModified = readRange(url, 0, 16)
    signature = 0x1A412743
    if struct.unpack("<I", header[:4])[0] == signature:
        endian = "<"
    elif struct.unpack(">I", header[:4])[0] == signature:
        endian = ">"
    else:
        raise ValueError(f"Invalid .2bit signature: {url}")
    _, version, sequenceCount, reserved = struct.unpack(f"{endian}4I", header)
    if version != 0 or reserved != 0:
        raise ValueError(f"Unsupported .2bit header: {url}")
    indexBytes, _ = readRange(url, 16, sequenceCount * 260)
    offset = 0
    sequenceOffsets = {}
    for _ in range(sequenceCount):
        nameLength = indexBytes[offset]
        offset += 1
        name = indexBytes[offset:offset + nameLength].decode("ascii")
        offset += nameLength
        sequenceOffset = struct.unpack_from(f"{endian}I", indexBytes, offset)[0]
        offset += 4
        if name in sequenceOffsets:
            raise ValueError(f"Duplicate .2bit chromosome: {url}: {name}")
        sequenceOffsets[name] = sequenceOffset
    nonNCounts = {}
    chromosomeRecords = {}
    for chrom, physicalLength in chromSizes.items():
        if chrom not in sequenceOffsets:
            raise ValueError(f"Missing .2bit chromosome: {url}: {chrom}")
        sequenceOffset = sequenceOffsets[chrom]
        prefix, _ = readRange(url, sequenceOffset, 8)
        dnaSize, nBlockCount = struct.unpack(f"{endian}2I", prefix)
        if dnaSize != physicalLength:
            raise ValueError(
                f"Physical length differs: {assembly}: {chrom}: "
                f"{physicalLength} != {dnaSize}"
            )
        blocks, _ = readRange(url, sequenceOffset + 8, 8 * nBlockCount + 4)
        starts = struct.unpack_from(f"{endian}{nBlockCount}I", blocks)
        sizes = struct.unpack_from(f"{endian}{nBlockCount}I", blocks, 4 * nBlockCount)
        blockEnd = 0
        for blockStart, blockSize in zip(starts, sizes):
            if blockStart < blockEnd or blockSize <= 0 or blockStart + blockSize > dnaSize:
                raise ValueError(f"Invalid N block: {assembly}: {chrom}")
            blockEnd = blockStart + blockSize
        nBases = sum(sizes)
        nonNCounts[chrom] = dnaSize - nBases
        chromosomeRecords[chrom] = {
            "physicalLength": dnaSize,
            "nBlockCount": nBlockCount,
            "nBases": nBases,
            "nonNBases": dnaSize - nBases,
        }
    (resourceDir / f"{assembly}.nonN.sizes").write_text(
        "".join(f"{chrom}\t{count}\n" for chrom, count in nonNCounts.items())
    )
    return assembly, {
        "sourceURL": url,
        "lastModified": lastModified,
        "formatVersion": version,
        "chromSizesResource": sizesPath.name,
        "chromosomes": chromosomeRecords,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resourceDir", type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "consenrich" / "data",
    )
    parser.add_argument(
        "assemblies", nargs="*", default=["ce10", "ce11", "dm6", "hg19", "hg38", "mm10", "mm39"]
    )
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = dict(executor.map(lambda name: readAssembly(name, args.resourceDir), args.assemblies))
    provenance = {
        "generatedUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formatSpecification": "https://genome.ucsc.edu/FAQ/FAQformat.html#format7",
        "generator": "scripts/buildNonNResources.py",
        "assemblies": records,
    }
    (args.resourceDir / "nonN.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
