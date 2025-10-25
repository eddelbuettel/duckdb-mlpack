
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_utilities.hpp>
#include <mlpack_model_data.hpp>

namespace duckdb {

unique_ptr<FunctionData> MlpackAdaboostTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names);

void MlpackAdaboostTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

unique_ptr<FunctionData> MlpackAdaboostPredTableBind(ClientContext &context, TableFunctionBindInput &input,
													 vector<LogicalType> &return_types, vector<string> &names);

void MlpackAdaboostPredTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

} // namespace duckdb
