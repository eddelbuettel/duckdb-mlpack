
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_utilities.hpp>
#include <mlpack_model_data.hpp>

namespace duckdb {

unique_ptr<FunctionData> MlpackLinRegTableBind(ClientContext &context, TableFunctionBindInput &input,
											   vector<LogicalType> &return_types, vector<string> &names);
void MlpackLinRegTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

unique_ptr<FunctionData> MlpackLinRegPredTableBind(ClientContext &context, TableFunctionBindInput &input,
												   vector<LogicalType> &return_types, vector<string> &names);
void MlpackLinRegPredTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

}
