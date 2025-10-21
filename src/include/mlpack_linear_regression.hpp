
#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp>
#include <duckdb_to_armadillo.hpp>

namespace duckdb {

struct MlpackLinRegData : TableFunctionData {
    bool data_returned = false;
	std::string features{""};
	std::string labels{""};
	std::string parameters{""};
 	vector<LogicalType> return_types;
	vector<string> names;
};

unique_ptr<FunctionData> MlpackLinRegTableBind(ClientContext &context, TableFunctionBindInput &input,
											   vector<LogicalType> &return_types, vector<string> &names);

void MlpackLinRegTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

}
