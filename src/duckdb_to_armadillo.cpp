
// Simple matrix and vector accessors from (possibly temporary) tables
// For simplicity we assume 'features' are of type double, and labels are of type int64

#include "duckdb_to_armadillo.hpp"

namespace duckdb {

template<typename T>
arma::Mat<T> get_armadillo_matrix(ClientContext &context, std::string &table) {

	Connection con(*context.db);

	std::string query = std::string("SELECT * FROM ") + table + std::string(";");
	std::cout << "  query: " << query << std::endl;
	auto result = con.Query(query);

	idx_t n = result->RowCount(), k = result->ColumnCount();
	std::cout << "  expecting " << n << " by " << k << std::endl;

	arma::Mat<T> m(n, k);
	idx_t row = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t row_idx = 0; row_idx < result->RowCount(); ++row_idx) {
			arma::Row<T> r(k);
			for (idx_t col_idx = 0; col_idx < k; col_idx++) {
				r(col_idx) = result->GetValue(col_idx, row_idx).GetValue<T>();
			}
			m.row(row++) = r;
			// auto field0 = result->GetValue(0 /*field0Index*/, row_idx).GetValue<double>();
			// auto field1 = result->GetValue(1 /*field1Index*/, row_idx).GetValue<double>();
			// auto field2 = result->GetValue(2 /*field2Index*/, row_idx).GetValue<double>();
			// auto field3 = result->GetValue(3 /*field3Index*/, row_idx).GetValue<double>();
			//objects.emplace_back(field0, field1, field2, field3);
			//std::cout << row_idx << " : " << field0 << " " << field1 << " " << field2 << " " << field3 << std::endl;
		}
	}
    m.print("m");
	return m;
}

template<typename T>
arma::Row<T> get_armadillo_row(ClientContext &context, std::string &table) {
	arma::Mat<T> m = get_armadillo_matrix<T>(context, table);
	assert(m.n_cols == 1);
	arma::Row<T> r = m.col(0);
	r.print("row");
	return r;
}

}
