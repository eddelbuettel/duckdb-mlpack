
#include "mlpack_adaboost.hpp"

namespace duckdb {

// mlpack adaboost accessor

void MlpackAdaboostTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");

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

	using PerceptronType = mlpack::Perceptron<mlpack::SimpleWeightUpdate, mlpack::ZeroInitialization, arma::mat>;
	mlpack::AdaBoost<PerceptronType, arma::mat> a;
	int numClasses = arma::max(labelsvec) + 1;
	const int iterations = params.count("iterations") > 0 ? std::stoi(params["iterations"]) : 100;
	const double tolerance = params.count("tolerance") > 0 ? std::stod(params["tolerance"]) : 2e-10;
	const int perceptronIter = params.count("perceptronIter") > 0 ? std::stoi(params["perceptronIter"]) : 400;
	const bool silent = params.count("silent") > 0 ? (params["silent"] == "true" ? true : false) : false;

	double ztProduct = a.Train(dataset, labelsvec, numClasses, iterations, tolerance, perceptronIter);

	if (verbose)
		std::cout << SerializeObject<mlpack::AdaBoost<PerceptronType, arma::mat>>(a) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::AdaBoost<PerceptronType, arma::mat>>(a));

	arma::Row<size_t> predictedLabels;
	a.Classify(dataset, predictedLabels);
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

void MlpackAdaboostPredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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

	using PerceptronType = mlpack::Perceptron<mlpack::SimpleWeightUpdate, mlpack::ZeroInitialization, arma::mat>;
	mlpack::AdaBoost<PerceptronType, arma::mat> a;
	UnserializeObject<mlpack::AdaBoost<PerceptronType, arma::mat>>(model, a);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::Row<size_t> classifiedvalues(n);
	a.Classify(dataset, classifiedvalues);
	if (verbose)
		classifiedvalues.print("predicted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)classifiedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
