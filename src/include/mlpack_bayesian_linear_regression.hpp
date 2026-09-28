
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_utilities.hpp>
#include <mlpack_model_data.hpp>

namespace duckdb {

void MlpackBayesianLinearRegressionTrainTableFunction(ClientContext &context, TableFunctionInput &data_p,
                                                      DataChunk &output);

void MlpackBayesianLinearRegressionPredictTableFunction(ClientContext &context, TableFunctionInput &data_p,
                                                        DataChunk &output);

} // namespace duckdb
