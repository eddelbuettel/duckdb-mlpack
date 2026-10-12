
#include <mlpack_decision_tree.hpp>
#include <mlpack/methods/decision_tree.hpp>

namespace duckdb {

void MlpackDecisionTreeTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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

	const size_t numClasses =
	    params.count("numClasses") > 0 ? std::stoul(params["numClasses"]) : arma::max(labelsvec) + 1;
	const size_t minimumLeafSize = params.count("minimumLeafSize") > 0 ? std::stoul(params["minimumLeafSize"]) : 10;
	const double minimumGainSplit = params.count("minimumGainSplit") > 0 ? std::stod(params["minimumGainSplit"]) : 1e-7;
	const size_t maximumDepth = params.count("maximumDepth") > 0 ? std::stoul(params["maximumDepth"]) : 0;
	if (params.count("silent") > 0)
		silent = (params["silent"] == "true" ? true : false);

	mlpack::DecisionTree<> dt(dataset, labelsvec, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth);

	if (verbose)
		std::cout << SerializeObject<mlpack::DecisionTree<>>(dt) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::DecisionTree<>>(dt));

	// Predict the labels on the input data (no train/test here)
	arma::Row<size_t> predictedLabels;
	dt.Classify(dataset, predictedLabels);
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

void MlpackDecisionTreePredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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

	mlpack::DecisionTree<> dt;
	UnserializeObject<mlpack::DecisionTree<>>(model, dt);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::Row<size_t> classifiedvalues(n);
	dt.Classify(dataset, classifiedvalues);
	if (verbose)
		classifiedvalues.print("predicted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)classifiedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
