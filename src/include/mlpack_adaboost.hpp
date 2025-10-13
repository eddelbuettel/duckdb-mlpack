
#pragma once

#include "duckdb.hpp"
#include <duckdb/storage/object_cache.hpp>

#include <mlpack.hpp>					// mlpack

namespace duckdb {

struct MlAdaboostState : public GlobalTableFunctionState {
	// nothing here as currently do not need state
};

struct MlAdaboostData : TableFunctionData {
	string key;
 	vector<LogicalType> return_types;
	vector<string> names;
};

unique_ptr<GlobalTableFunctionState> MlAdaboostGlobalInit(ClientContext &context, TableFunctionInitInput &input);

unique_ptr<LocalTableFunctionState> MlAdaboostLocalInit(ExecutionContext &context, TableFunctionInitInput &data_p, GlobalTableFunctionState *global_state);

unique_ptr<FunctionData> MlAdaboostTableBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);

OperatorResultType MlAdaboostFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input, DataChunk &output);

OperatorFinalizeResultType MlAdaboostFinaliseFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &outdata_p);

}
