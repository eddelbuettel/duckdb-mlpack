#define DUCKDB_EXTENSION_MAIN

#include "mlpack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include <openssl/opensslv.h>			// OpenSSL linked through vcpkg
#include <mlpack.hpp>					// mlpack

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

inline void MlpackMlpackVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack " + name.GetString() + ", my included mlpack version is " + mlpack::util::GetVersion() );
        });
}

// Table code below

static bool verbose = false;		// toggle for debug / verbosity messages

struct MlpackTableData : public TableFunctionData {
    bool data_returned = false;  // Add this flag
	int value = 0;				 // for passed in matrix start value
    vector<Value> col1;
    vector<Value> col2;
    vector<Value> col3;
    vector<Value> col4;
    vector<Value> col5;

    MlpackTableData() {}
};

static unique_ptr<FunctionData> MlpackTableBind(ClientContext &context, TableFunctionBindInput &input,
												vector<LogicalType> &return_types, vector<string> &names) {

	auto data = make_uniq<MlpackTableData>();

	if (verbose) std::cout << "MlpackTableBind\n";
	if (verbose) std::cout << "  old value=" << data->value << std::endl;
	data->value = input.inputs[0].GetValue<int>();
	if (verbose) std::cout << "  setting value=" << data->value << std::endl;

    names = { "col_1", "col_2", "col_3", "col_4", "col_5" };
    return_types = { LogicalType::SMALLINT, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::DOUBLE };

    return std::move(data);
}

static void MlpackTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	if (verbose) std::cout << "MlpackTableFunction\n";
	auto &data = const_cast<MlpackTableData&>(data_p.bind_data->Cast<MlpackTableData>());

	// if we have been called, return nothing
    if (data.data_returned) {
        output.SetCardinality(0);
        return;
    }
	if (verbose) std::cout << "  seeing value=" << data.value << std::endl;

	idx_t chunk_size = 3; // arbitrary

    output.SetCardinality(chunk_size);

	int32_t val = data.value;

    for (idx_t i = 0; i < chunk_size; i++) {
        output.data[0].SetValue(i, val++);
        output.data[1].SetValue(i, val++);
        output.data[2].SetValue(i, val++);
        output.data[3].SetValue(i, val++);
        output.data[4].SetValue(i, val++);
    }

	// mark that we have been called
	data.data_returned = true;
}

static void LoadInternal(DatabaseInstance &instance) {
    // Register a scalar function
    auto mlpack_scalar_function = ScalarFunction("mlpack", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlpackScalarFun);
    ExtensionUtil::RegisterFunction(instance, mlpack_scalar_function);

    // Register another scalar function
    auto mlpack_openssl_version_scalar_function = ScalarFunction("mlpack_openssl_version", {LogicalType::VARCHAR},
																 LogicalType::VARCHAR, MlpackOpenSSLVersionScalarFun);
    ExtensionUtil::RegisterFunction(instance, mlpack_openssl_version_scalar_function);

    // Register another scalar function
    auto mlpack_mlpack_version_scalar_function = ScalarFunction("mlpack_mlpack_version", {LogicalType::VARCHAR},
																LogicalType::VARCHAR, MlpackMlpackVersionScalarFun);
    ExtensionUtil::RegisterFunction(instance, mlpack_mlpack_version_scalar_function);

	// Register sample table function
	auto mlpack_sample_table_function = TableFunction("mlpack_table", {LogicalType::INTEGER}, MlpackTableFunction, MlpackTableBind);
	ExtensionUtil::RegisterFunction(instance, mlpack_sample_table_function);
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
