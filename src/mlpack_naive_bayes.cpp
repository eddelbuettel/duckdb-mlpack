
#include <mlpack_naive_bayes.hpp>
#include <mlpack/methods/naive_bayes.hpp>

namespace duckdb {

void MlpackNaiveBayesTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");
	bool silent = get_setting<bool>(context, "mlpack_silent");

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

	const int numClasses = params.count("numClasses") > 0 ? std::stoi(params["numClasses"]) : arma::max(labelsvec) + 1;
	const bool incrementalVariance =
	    params.count("incremental_variance") > 0 ? (params["incremental_variance"] == "true" ? true : false) : false;
	if (params.count("silent") > 0)
		silent = (params["silent"] == "true" ? true : false);

	mlpack::NaiveBayesClassifier<> nbc(dataset, labelsvec, numClasses, incrementalVariance);

	if (verbose)
		std::cout << SerializeObject<mlpack::NaiveBayesClassifier<>>(nbc) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::NaiveBayesClassifier<>>(nbc));

	// Predict the labels on the input data (no train/test here)
	arma::Row<size_t> predictedLabels;
	nbc.Classify(dataset, predictedLabels);
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
}

void MlpackNaiveBayesPredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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

	mlpack::NaiveBayesClassifier<> nbc;
	UnserializeObject<mlpack::NaiveBayesClassifier<>>(model, nbc);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::Row<size_t> classifiedvalues(n);
	nbc.Classify(dataset, classifiedvalues);
	if (verbose)
		classifiedvalues.print("predicted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)classifiedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
