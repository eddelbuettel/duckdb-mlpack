#define DUCKDB_EXTENSION_MAIN

#include "mlpack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

inline void MlpackScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack "+name.GetString()+" 🐥");;
        });
}

inline void MlpackOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack " + name.GetString() +
                                                     ", my linked OpenSSL version is " +
                                                     OPENSSL_VERSION_TEXT );;
        });
}

static void LoadInternal(DatabaseInstance &instance) {
    // Register a scalar function
    auto mlpack_scalar_function = ScalarFunction("mlpack", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlpackScalarFun);
    ExtensionUtil::RegisterFunction(instance, mlpack_scalar_function);

    // Register another scalar function
    auto mlpack_openssl_version_scalar_function = ScalarFunction("mlpack_openssl_version", {LogicalType::VARCHAR},
                                                LogicalType::VARCHAR, MlpackOpenSSLVersionScalarFun);
    ExtensionUtil::RegisterFunction(instance, mlpack_openssl_version_scalar_function);
}

void MlpackExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
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

DUCKDB_EXTENSION_API void mlpack_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::MlpackExtension>();
}

DUCKDB_EXTENSION_API const char *mlpack_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
