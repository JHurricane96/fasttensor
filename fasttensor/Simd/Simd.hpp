#pragma once

#include "SimdMacros.hpp"

#if SSE_INSTR_SET > 7
#  include "Avx2.hpp"
#else
#  include "Generic.hpp"
#endif
