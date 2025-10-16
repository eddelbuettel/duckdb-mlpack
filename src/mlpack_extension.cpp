#define DUCKDB_EXTENSION_MAIN

#include "mlpack_extension.hpp"

#include "mlpack_table_function.hpp"
#include "mlpack_adaboost.hpp"

#include <duckdb.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include <mlpack.hpp>					// mlpack

namespace duckdb {

// Function loading

static void LoadInternal(ExtensionLoader &loader) {
	// Register sample table function returning a table (and consuming a scalar (or two, commented out)
	auto mlpack_sample_table_function = TableFunction("mlpack_table", {LogicalType::INTEGER /*, LogicalType::VARCHAR */}, MlpackTableFunction, MlpackTableBind);
	loader.RegisterFunction(mlpack_sample_table_function);

	// Register adaboost example
	{
		TableFunction mlpack_adaboost_function("mlpack_adaboost",
											   { LogicalType::TABLE },
											   nullptr /*MlAdaboostTableFunction*/,
											   MlAdaboostTableBind,
											   MlAdaboostGlobalInit,
											   MlAdaboostLocalInit);
		mlpack_adaboost_function.in_out_function = MlAdaboostFunction;
		mlpack_adaboost_function.in_out_function_final = MlAdaboostFinaliseFunction;
		loader.RegisterFunction(mlpack_adaboost_function);
	}
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
