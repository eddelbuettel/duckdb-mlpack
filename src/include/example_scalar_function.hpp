#pragma once

#include "duckdb.hpp"

namespace duckdb {

	void MlpackScalarFun(DataChunk &args, ExpressionState &state, Vector &result);

}
