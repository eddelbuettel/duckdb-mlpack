#pragma once

#include <duckdb.hpp>

namespace duckdb {

unique_ptr<FunctionData> MlpackTrainTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names);

unique_ptr<FunctionData> MlpackPredictTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names);

} // namespace duckdb
