
// Simple matrix and vector accessors from (possibly temporary) tables
// For simplicity we assume 'features' are of type double, and labels are of type int64

#include <duckdb_utilities.hpp>

namespace duckdb {

std::map<std::string, std::string> get_parameters(ClientContext &context, std::string parameters) {
	Connection con(*context.db);

	std::string query = std::string("SELECT * FROM ") + parameters + std::string(";");
	auto result = con.Query(query);

	idx_t n = result->RowCount(), k = result->ColumnCount();
	std::map<std::string, std::string> params;

	while (auto chunk = result->Fetch()) {
		for (idx_t row_idx = 0; row_idx < n; row_idx++) {
			std::string key = result->GetValue(0, row_idx).GetValue<std::string>();
			std::string val = result->GetValue(1, row_idx).GetValue<std::string>();
			//std::cout << key << " -> " << val << std::endl;
			params.try_emplace(key, val);
		}
	}
	return params;
}

	void store_model(ClientContext &context, std::string model_table, std::string model_as_json) {
	Connection con(*context.db);

	std::string query = std::string("INSERT INTO " + model_table + " VALUES ('" + model_as_json + "');");
	con.Query(query);
}

}
