#pragma once

#include <duckdb.hpp>

namespace duckdb {

// returning 'int' for classification
unique_ptr<FunctionData> MlpackTrainTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names);
// three inputs for unsupervisied training such as k-means: given X but no Y
unique_ptr<FunctionData> MlpackUnsupervisedTrainTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                             vector<LogicalType> &return_types, vector<string> &names);

unique_ptr<FunctionData> MlpackPredictTableBindInt(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names);

// returning 'double' for regression
unique_ptr<FunctionData> MlpackTrainTableBindDouble(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names);

unique_ptr<FunctionData> MlpackPredictTableBindDouble(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names);

void MlpackMlpackVersion(DataChunk &args, ExpressionState &state, Vector &result);
void MlpackArmadilloVersion(DataChunk &args, ExpressionState &state, Vector &result);

} // namespace duckdb
