
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_to_armadillo.hpp>

namespace duckdb {

struct MlAdaboostData : TableFunctionData {
    bool data_returned = false;
	std::string features{""};
	std::string labels{""};
	std::string parameters{""};
 	vector<LogicalType> return_types;
	vector<string> names;
};

unique_ptr<FunctionData> MlpackAdaboostTableBind(ClientContext &context, TableFunctionBindInput &input,
										 vector<LogicalType> &return_types, vector<string> &names);

void MlpackAdaboostTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

}
