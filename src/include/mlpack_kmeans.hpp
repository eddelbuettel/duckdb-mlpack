
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_utilities.hpp>
#include <mlpack_model_data.hpp>

namespace duckdb {

void MlpackKmeansTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

//void MlpackKmeansPredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

} // namespace duckdb
