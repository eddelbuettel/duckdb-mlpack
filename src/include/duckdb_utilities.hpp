
// Simple matrix and vector accessors from (possibly temporary) tables
// For simplicity we assume 'features' are of type double, and labels are of type int64

#pragma once

#include <duckdb.hpp>
#include <mlpack.hpp> // mlpack

namespace duckdb {

std::map<std::string, std::string> get_parameters(ClientContext &context, std::string parameters);
void store_model(ClientContext &context, std::string model_table, std::string model_as_json);
std::string retrieve_model(ClientContext &context, std::string model_table);
std::string serialize_vector(const arma::vec &vec);
void store_vector(ClientContext &context, std::string model_table, std::string key, std::string model_as_json);

template <typename T>
arma::Mat<T> get_armadillo_matrix_transposed(ClientContext &context, std::string &table) {

	Connection con(*context.db);

	std::string query = std::string("SELECT * FROM ") + table + std::string(";");
	auto result = con.Query(query);

	idx_t n = result->RowCount(), k = result->ColumnCount();
	// std::cout << "TOTAL: " << n << " by " << k << std::endl;
	arma::Mat<T> m(k, n); // transpose while reading, i.e. row i becomes col i
	idx_t row = 0;
	while (auto chunk = result->Fetch() && row < n) {
		for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
			arma::Col<T> r(k);
			for (idx_t col_idx = 0; col_idx < k; col_idx++) {
				r(col_idx) = result->GetValue(col_idx, row_idx).GetValue<T>();
			}
			m.col(row++) = r;
		}
	}
	// m.print("m");
	return m;
}

template <typename T>
arma::Row<T> get_armadillo_row(ClientContext &context, std::string &table) {
	arma::Mat<T> m = get_armadillo_matrix_transposed<T>(context, table);
	assert(m.n_rows == 1);
	arma::Row<T> r = m.row(0);
	// r.print("row");
	return r;
}

template <typename T>
std::string SerializeObject(T &t) {
	std::ostringstream oss;
	{ // need a block to have 'o' destructed for final closing '}'
		cereal::JSONOutputArchive o(oss);
		T &x(t);
		o(CEREAL_NVP(x));
	}
	return oss.str();
};

template <typename T>
void UnserializeObject(std::string json, T &t) {
	std::istringstream iss(json);
	cereal::JSONInputArchive i(iss);
	T &x(t);
	i(CEREAL_NVP(x));
};

template <typename T>
T get_setting(ClientContext &context, std::string key) {
	Value mlpack_value;
	context.TryGetCurrentSetting(key, mlpack_value);
	return mlpack_value.GetValue<T>();
}

} // namespace duckdb
