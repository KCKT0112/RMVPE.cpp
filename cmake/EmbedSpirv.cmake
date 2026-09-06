# SPDX-License-Identifier: MPL-2.0
# Adapted from KCKT0112/FCPE.cpp.
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
file(WRITE "${OUTPUT}" "// Generated SPIR-V.\nalignas(4) static const unsigned char rmvpe_gru_spv[] = {${bytes}};\n")
