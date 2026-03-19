Consenrich is distributed under the MIT license in [LICENSE](LICENSE)

This distribution *vendors* source from [HTSlib](https://www.htslib.org/)

The following are included:

- HTSlib
  - source path: `vendor/htslib`
  - upstream license file: `vendor/htslib/LICENSE`
  - packaged license copy: `HTSLIB_LICENSE.txt`
  - license summary: files outside `cram/` are MIT/Expat and files in `cram/`
    are under the modified 3-clause BSD license

- htscodecs (CRAM)
  - source path: `vendor/htslib/htscodecs`
  - upstream license file: `vendor/htslib/htscodecs/LICENSE.md`
  - packaged license copy: `HTSCODECS_LICENSE.md`
  - license summary: BSD-style license with some files noted as public domain
    or CC0-derived in the upstream license text

**Please NOTE**

- Consenrich -- keeps its own MIT license
- The bundled third-party code retains its original licenses
- *Source and binary redistributions should preserve the upstream copyright
  notices, license terms, and disclaimers*

For reference, here are citations for manuscripts corresponding to the bundled code:

```bibtex
@article{HTSlib2021,
    author = {Bonfield, James K and Marshall, John and Danecek, Petr and Li, Heng and Ohan, Valeriu and Whitwham, Andrew and Keane, Thomas and Davies, Robert M},
    title = "{HTSlib: C library for reading/writing high-throughput sequencing data}",
    journal = {GigaScience},
    volume = {10},
    number = {2},
    year = {2021},
    month = {02},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giab007},
    url = {https://doi.org/10.1093/gigascience/giab007},
    note = {giab007},
    eprint = {https://academic.oup.com/gigascience/article-pdf/10/2/giab007/36332285/giab007.pdf},
}

