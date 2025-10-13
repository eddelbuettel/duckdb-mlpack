
#include "mlpack_adaboost.hpp"

namespace duckdb {

static bool verbose = false;

// mlpack adaboost accessor

unique_ptr<GlobalTableFunctionState> MlAdaboostGlobalInit(ClientContext &context, TableFunctionInitInput &input) {
	if (verbose) std::cout << "MlAdaboostGlobalInit\n";
 	return make_uniq<MlAdaboostState>();
}

unique_ptr<LocalTableFunctionState> MlAdaboostLocalInit(ExecutionContext &context, TableFunctionInitInput &data_p,
															   GlobalTableFunctionState *global_state) {
	if (verbose) std::cout << "MlAdaboostLocalInit\n";
 	//state.currently_adding++;
 	return make_uniq<LocalTableFunctionState>();
}

unique_ptr<FunctionData> MlAdaboostTableBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names) {
	if (verbose) std::cout << "MlAdaboostTableBind\n";
	auto resdata = make_uniq<MlAdaboostData>(); // 'resdata' for result data i.e. outgoing
	resdata->key = string("the quick brown fox");
    names = { "predicted" };
    return_types = { LogicalType::INTEGER };
	resdata->return_types = return_types;
	resdata->names = names;
    return std::move(resdata);
}

OperatorResultType MlAdaboostFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input, DataChunk &output) {
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

OperatorFinalizeResultType MlAdaboostFinaliseFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &outdata_p) {
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

}
