#pragma once

#include <duckdb.hpp>

namespace duckdb {

struct MlpackModelData : TableFunctionData {
	bool data_returned = false;
	std::string features {""};
	std::string labels {""};
	std::string parameters {""};
	std::string model {""};
	vector<LogicalType> return_types;
	vector<string> names;
};

} // namespace duckdb
