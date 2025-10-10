#define DUCKDB_EXTENSION_MAIN

//#define ARMA_DONT_USE_WRAPPER 1
#include "mlpack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/storage/object_cache.hpp"

#include <openssl/opensslv.h>			// OpenSSL linked through vcpkg
#include <mlpack.hpp>					// mlpack

namespace duckdb {

// Example scalar function
inline void MlpackScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack " + name.GetString() + " 🐥");;
        });
}

// Example scalar function accessing another header
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

// Example scalar function accessing mlpack header
inline void MlpackMlpackVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "Mlpack " + name.GetString() + ", my included mlpack version is " + mlpack::util::GetVersion() );
        });
}


// (Sample) Table code below -- for sample export of 5 columns with one simple scalar as param

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

	auto resdata = make_uniq<MlpackTableData>(); // 'resdata' for result data i.e. outgoing

	if (verbose) std::cout << "MlpackTableBind\n";
	if (verbose) std::cout << "  old value=" << resdata->value << std::endl;
	resdata->value = input.inputs[0].GetValue<int>();
	// if there was a second VARCHAR arg:  std::string tempval = input.inputs[1].GetValue<std::string>();
	if (verbose) std::cout << "  setting value=" << resdata->value << std::endl;

    names = { "col_1", "col_2", "col_3", "col_4", "col_5" };
    return_types = { LogicalType::SMALLINT, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::FLOAT, LogicalType::DOUBLE };

    return std::move(resdata);
}

static void MlpackTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	if (verbose) std::cout << "MlpackTableFunction\n";
	auto &resdata = const_cast<MlpackTableData&>(data_p.bind_data->Cast<MlpackTableData>());

	// if we have been called, return nothing
    if (resdata.data_returned) {
        output.SetCardinality(0);
        return;
    }
	if (verbose) std::cout << "  seeing value=" << resdata.value << std::endl;

	idx_t chunk_size = 3; // arbitrary

    output.SetCardinality(chunk_size);

	int32_t val = resdata.value;

    for (idx_t i = 0; i < chunk_size; i++) {
        output.data[0].SetValue(i, val++);
        output.data[1].SetValue(i, val++);
        output.data[2].SetValue(i, val++);
        output.data[3].SetValue(i, val++);
        output.data[4].SetValue(i, val++);
    }

	// mark that we have been called
	resdata.data_returned = true;
}

// mlpack adaboost accessor

struct MlAdaboostState : public GlobalTableFunctionState {
	// nothing here as currently do not need state
};

struct MlAdaboostData : TableFunctionData {
	string key;
 	vector<LogicalType> return_types;
	vector<string> names;
};

static unique_ptr<GlobalTableFunctionState> MlAdaboostGlobalInit(ClientContext &context, TableFunctionInitInput &input) {
	if (verbose) std::cout << "MlAdaboostGlobalInit\n";
 	return make_uniq<MlAdaboostState>();
}

static unique_ptr<LocalTableFunctionState> MlAdaboostLocalInit(ExecutionContext &context, TableFunctionInitInput &data_p,
															   GlobalTableFunctionState *global_state) {
	if (verbose) std::cout << "MlAdaboostLocalInit\n";
 	//state.currently_adding++;
 	return make_uniq<LocalTableFunctionState>();
}

static unique_ptr<FunctionData> MlAdaboostTableBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names) {
	if (verbose) std::cout << "MlAdaboostTableBind\n";
	auto resdata = make_uniq<MlAdaboostData>(); // 'resdata' for result data i.e. outgoing
	resdata->key = string("the quick brown fox");
    names = { "predicted" };
    return_types = { LogicalType::INTEGER };
	resdata->return_types = return_types;
	resdata->names = names;
    return std::move(resdata);
}


static OperatorResultType MlAdaboostFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input, DataChunk &output) {
	if (verbose) std::cout << "MlAdaboostFunction\n";
 	//MlAdaboostState &state = global_state->Cast<MlAdaboostState>();
	auto bind_data = data_p.bind_data->Cast<MlAdaboostData>();
	auto &object_cache = ObjectCache::GetObjectCache(context.client);
	//std::cout << "key is " << bind_data.key << std::endl;
	auto n = input.size();
	auto k = input.ColumnCount();
	if (verbose) {
		std::cout << "  input size: " << input.size() << std::endl;
		std::cout << "  col count: " << input.ColumnCount() << std::endl;
		std::cout << "  capacity: " << input.GetCapacity() << std::endl;
		std::cout << "  string: " << input.ToString() << std::endl;
	}
	arma::mat dataset(n, k);
	for (auto i=0; i<k; i++) {
		auto& vector = input.data[i];
		auto data_ptr = reinterpret_cast<double*>(vector.GetData());
		arma::rowvec v(data_ptr, n, false, true);
		if (verbose) v.print("col" + std::to_string(i));
		dataset.col(i) = v.t();
	}
	if (verbose) dataset.print("dataset");

	// we now have a matrix m and its last column is the labels, let's extract it
	arma::Row<size_t> labelsvec = arma::conv_to<arma::Row<size_t>>::from(dataset.col(k-1));
	if (verbose) labelsvec.print("labels");
	// and shed it, and tranpose
	dataset.shed_col(k-1);
	dataset = dataset.t();
	if (verbose) dataset.print("dataset");

    using PerceptronType = mlpack::Perceptron<mlpack::SimpleWeightUpdate, mlpack::ZeroInitialization, arma::mat>;
    mlpack::AdaBoost<PerceptronType, arma::mat> a;
    int numClasses = arma::max(labelsvec) + 1;
	// these could be / should parameters
	constexpr int iterations = 100;
	constexpr double tolerance = 2e-10;
	constexpr int perceptronIter = 400;

    double ztProduct = a.Train(dataset, labelsvec, numClasses, iterations, tolerance, perceptronIter);

    arma::Row<size_t> predictedLabels;
    a.Classify(dataset, predictedLabels);
	if (verbose) predictedLabels.print("predicted");
    size_t countError = arma::accu(labelsvec != predictedLabels);
	std::cout << "Misclassified: " << countError << std::endl;

	output.SetCardinality(n);

	auto &input_vector = input.data[0];  // somewhat spurious init just to get a vector

	// Prepare first (sole) column of output
    auto &result_vector = output.data[0];

    // Use UnifiedVectorFormat for efficient access
    UnifiedVectorFormat input_data;
    input_vector.ToUnifiedFormat(n, input_data);

    auto input_ptr = UnifiedVectorFormat::GetData<double>(input_data);
    auto result_data = FlatVector::GetData<int>(result_vector);
    auto &result_validity = FlatVector::Validity(result_vector);

	// Process each element
    for (idx_t i = 0; i < n; i++) {
        auto idx = input_data.sel->get_index(i);
        // check if valid (not NULL)
        if (!input_data.validity.RowIsValid(idx)) {
            result_validity.SetInvalid(i);
            continue;
        }
		// copy -- would should be able to do better but let's be pedestrian at first
        result_data[i] = predictedLabels[i];
    }

	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType MlAdaboostFinaliseFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &outdata_p) {
	if (verbose) std::cout << "MlAdaboostFinaliseFunction\n";
	return OperatorFinalizeResultType::FINISHED;
}

// static void MlAdaboostTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
// 	if (verbose) std::cout << "MlAdaboostTableFunction\n";

// 	//if (verbose) std::cout << "MlpackTableFunction\n";
// 	auto &resdata = const_cast<MlpackTableData&>(data_p.bind_data->Cast<MlpackTableData>());

// 	// if we have been called, return nothing
//     if (resdata.data_returned) {
//         output.SetCardinality(0);
//         return;
//     }
// 	if (verbose) std::cout << "  seeing value=" << resdata.value << std::endl;

// 	idx_t chunk_size = 3; // arbitrary

//     output.SetCardinality(chunk_size);

// 	int32_t val = resdata.value;

//     for (idx_t i = 0; i < chunk_size; i++) {
//         output.data[0].SetValue(i, val++);
//         output.data[1].SetValue(i, val++);
//         output.data[2].SetValue(i, val++);
//         output.data[3].SetValue(i, val++);
//         output.data[4].SetValue(i, val++);
//     }

// 	// mark that we have been called
// 	resdata.data_returned = true;

// 	//auto &resdata = const_cast<MlpackTableData&>(data_p.bind_data->Cast<MlpackTableData>());
// 	//resdata.data_returned = true;	// mark that we have been called
// }

// Function loading

static void LoadInternal(ExtensionLoader &loader) {
    // Register a scalar function
    auto mlpack_scalar_function = ScalarFunction("mlpack", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlpackScalarFun);
    loader.RegisterFunction(mlpack_scalar_function);

    // Register another scalar function
    auto mlpack_openssl_version_scalar_function =
		ScalarFunction("mlpack_openssl_version", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlpackOpenSSLVersionScalarFun);
    loader.RegisterFunction(mlpack_openssl_version_scalar_function);

    // Register another scalar function
    auto mlpack_mlpack_version_scalar_function =
		ScalarFunction("mlpack_mlpack_version", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlpackMlpackVersionScalarFun);
    loader.RegisterFunction(mlpack_mlpack_version_scalar_function);

	// Register sample table function returning a table (and consuming a scalar (or two, commented out)
	auto mlpack_sample_table_function = TableFunction("mlpack_table", {LogicalType::INTEGER /*, LogicalType::VARCHAR */}, MlpackTableFunction, MlpackTableBind);
	loader.RegisterFunction(mlpack_sample_table_function);

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
