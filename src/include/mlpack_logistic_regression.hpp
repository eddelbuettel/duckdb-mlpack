
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_utilities.hpp>
#include <mlpack_model_data.hpp>

namespace duckdb {

void MlpackLogisticRegTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

void MlpackLogisticRegPredTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

} // namespace duckdb
