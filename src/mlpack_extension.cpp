#define DUCKDB_EXTENSION_MAIN

#include "mlpack_extension.hpp"

#include "mlpack_table_function.hpp"
#include "mlpack_adaboost.hpp"
#include "mlpack_linearregression.hpp"

#include <duckdb.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include <mlpack.hpp>					// mlpack

namespace duckdb {

// Function loading

static void LoadInternal(ExtensionLoader &loader) {
	auto &dbinstance = loader.GetDatabaseInstance();
	Connection con(dbinstance);

	// Register sample table function returning a table (and consuming a scalar (or two, commented out)
	auto mlpack_sample_table_function = TableFunction("mlpack_table", { LogicalType::INTEGER }, MlpackTableFunction, MlpackTableBind);
	loader.RegisterFunction(mlpack_sample_table_function);

	// Register adaboost example
	auto mlpack_adaboost_function = TableFunction("mlpack_adaboost", { LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR }, MlpackAdaboostTableFunction, MlpackAdaboostTableBind);
	loader.RegisterFunction(mlpack_adaboost_function);

	// Register linear regression example
	auto mlpack_linreg_function = TableFunction("mlpack_linearregression", { LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR }, MlpackLinRegTableFunction, MlpackLinRegTableBind);
	loader.RegisterFunction(mlpack_linreg_function);
}

void MlpackExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string MlpackExtension::Name() {
	return "mlpack";
}

std::string MlpackExtension::Version() const {
#ifdef EXT_VERSION_MLPACK
	return EXT_VERSION_MLPACK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

	DUCKDB_CPP_EXTENSION_ENTRY(mlpack, loader) {
		duckdb::LoadInternal(loader);
	}

	DUCKDB_EXTENSION_API const char *mlpack_version() {
		return duckdb::DuckDB::LibraryVersion();
	}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
