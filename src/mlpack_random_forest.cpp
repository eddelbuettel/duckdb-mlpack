
#include "mlpack_random_forest.hpp"

namespace duckdb {

// mlpack adaboost accessor

void MlpackRandomForestTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");
	bool silent = get_setting<bool>(context, "mlpack_silent");

	if (verbose)
		std::cout << "MlpackAdaboostFunction\n";
	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetCardinality(0);
		if (verbose)
			std::cout << "  done\n";
		return;
	}

	// Explanatory variables i.e. 'features'
	arma::mat dataset = get_armadillo_matrix_transposed<double>(context, resdata.features);
	if (verbose)
		dataset.print("dataset");
	// Dependent variable i.e. 'labels' (as double because of macos arm64 linker)
	arma::Row<double> labelsvecdbl = get_armadillo_row<double>(context, resdata.labels);
	arma::Row<size_t> labelsvec = arma::conv_to<arma::Row<size_t>>::from(labelsvecdbl);
	if (arma::min(labelsvec) != 0)
		labelsvec = labelsvec - 1;
	if (verbose)
		labelsvec.print("labelsvec");
	std::map<std::string, std::string> params = get_parameters(context, resdata.parameters);

	int numClasses = arma::max(labelsvec) + 1;
	const int nclasses = params.count("nclasses") > 0 ? std::stoi(params["nclasses"]) : numClasses;
	const int ntrees = params.count("ntrees") > 0 ? std::stoi(params["ntrees"]) : 20;
	if (params.count("silent") > 0) silent = (params["silent"] == "true" ? true : false);
	const size_t seed = params.count("seed") > 0 ? std::stoi(params["seed"]) : -1;
	const int threads = params.count("threads") > 0 ? std::stoi(params["threads"]) : -1;
	int curr_num_threads = -1;

#pragma omp parallel
	{ curr_num_threads = omp_get_num_threads(); }
	if (threads != -1) {              // && curr_num_threads != 1) {
		omp_set_num_threads(threads); // for the number of threads to one
		if (verbose) {
			std::cout << "Setting threads from " << curr_num_threads << " to " << threads << std::endl;
		}
	}
	if (seed != -1) {
		mlpack::RandomSeed(seed);
		if (verbose) {
			std::cout << "Setting seed " << seed << std::endl;
		}
	}

	mlpack::RandomForest rf(dataset, labelsvec, nclasses, ntrees);

	if (verbose)
		std::cout << SerializeObject<mlpack::RandomForest<>>(rf) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::RandomForest<>>(rf));

	// Predict the labels on the input data (no train/test here)
	arma::Row<size_t> predictedLabels;
	rf.Classify(dataset, predictedLabels);
	if (verbose)
		predictedLabels.print("predicted");
	size_t countError = arma::accu(labelsvec != predictedLabels);
	if (!silent)
		std::cout << "Misclassified: " << countError << std::endl;

	auto n = predictedLabels.n_elem;
	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int)predictedLabels[i]);
	}

	resdata.data_returned = true; // mark that we have been called

	if (threads != curr_num_threads) {
		omp_set_num_threads(curr_num_threads); // for the number of threads to one
		if (verbose) {
			std::cout << "Resetting threads to " << curr_num_threads << std::endl;
		}
	}
}

void MlpackRandomForestPredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");
	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetCardinality(0);
		if (verbose)
			std::cout << "  done\n";
		return;
	}

	// Explanatory variables i.e. 'features'
	arma::mat dataset = get_armadillo_matrix_transposed<double>(context, resdata.features);
	if (verbose)
		dataset.print("dataset");

	auto model = retrieve_model(context, resdata.model);
	if (verbose)
		std::cout << model << std::endl;

	mlpack::RandomForest rf;
	UnserializeObject<mlpack::RandomForest<>>(model, rf);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::Row<size_t> classifiedvalues(n);
	rf.Classify(dataset, classifiedvalues);
	if (verbose)
		classifiedvalues.print("predicted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)classifiedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
