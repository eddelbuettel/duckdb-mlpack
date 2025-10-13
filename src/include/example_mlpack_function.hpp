#pragma once

#include "duckdb.hpp"

#include <mlpack.hpp>					// mlpack

namespace duckdb {

	void MlpackMlpackVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result);

}
