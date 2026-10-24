#include <mlpack_linear_svm.hpp>
#include <mlpack/methods/linear_svm.hpp>

namespace duckdb {

void MlpackLinearSVMTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");
	bool silent = get_setting<bool>(context, "mlpack_silent");

	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetChildCardinality(0);
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
	const double lambda = params.count("lambda") > 0 ? std::stod(params["lambda"]) : 0.0001;
	const double delta = params.count("delta") > 0 ? std::stod(params["delta"]) : 1.0;
	const bool fitIntercept =
	    params.count("fitIntercept") > 0 ? (params["fitIntercept"] == "true" ? true : false) : true;
	const bool standardize = params.count("standardize") > 0 ? (params["standardize"] == "true" ? true : false) : true;
	const size_t maxIterations = params.count("maxIterations") > 0 ? std::stoul(params["maxIterations"]) : 10000;
	const double tolerance = params.count("tolerance") > 0 ? std::stod(params["tolerance"]) : 1e-10;
	if (params.count("silent") > 0)
		silent = (params["silent"] == "true" ? true : false);

	// Linear svm is sensitive to the scale of the features, so we standardize
	// the data by default (and retain the per-feature means and standard
	// deviations for the same transform at prediction time)
	if (standardize) {
		arma::vec means = arma::mean(dataset, 1);     // per-feature means
		arma::vec stds = arma::stddev(dataset, 0, 1); // per-feature (population) standard deviations
		stds.elem(arma::find(stds < 1e-12)).ones();   // avoid division by zero for constant features
		dataset.each_col() -= means;
		dataset.each_col() /= stds;
		store_vector(context, resdata.model, "means", serialize_vector(means));
		store_vector(context, resdata.model, "stds", serialize_vector(stds));
	}

	mlpack::LinearSVM<> lsvm;
	lsvm.Lambda() = lambda;
	lsvm.Delta() = delta;
	lsvm.NumClasses() = numClasses;
	lsvm.FitIntercept() = fitIntercept;
	ens::L_BFGS lbfgs;
	lbfgs.MaxIterations() = maxIterations;
	lbfgs.MinGradientNorm() = tolerance;
	if (verbose)
		std::cout << "Training model with L-BFGS optimizer." << std::endl;
	lsvm.Train(dataset, labelsvec, numClasses, lbfgs);

	if (verbose)
		std::cout << SerializeObject<mlpack::LinearSVM<>>(lsvm) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::LinearSVM<>>(lsvm));

	// Predict the labels on the input data (no train/test here)
	arma::Row<size_t> predictedLabels;
	lsvm.Classify(dataset, predictedLabels);
	if (verbose)
		predictedLabels.print("predicted");
	size_t countError = arma::accu(labelsvec != predictedLabels);
	if (!silent)
		std::cout << "Misclassified: " << countError << std::endl;

	auto n = predictedLabels.n_elem;
	output.SetChildCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int)predictedLabels[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

void MlpackLinearSVMPredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");

	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetChildCardinality(0);
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

	// if the model was fit with standardized features, apply the same transform
	auto means = retrieve_vector(context, resdata.model, "means");
	if (!means.empty()) {
		std::vector<double> meanvec, stdvec;
		{
			std::istringstream iss(means);
			cereal::JSONInputArchive i(iss);
			i(cereal::make_nvp("parameters", meanvec));
		}
		auto stds = retrieve_vector(context, resdata.model, "stds");
		{
			std::istringstream iss(stds);
			cereal::JSONInputArchive i(iss);
			i(cereal::make_nvp("parameters", stdvec));
		}
		for (idx_t row = 0; row < dataset.n_rows; row++)
			dataset.row(row) = (dataset.row(row) - meanvec[row]) / stdvec[row];
	}

	mlpack::LinearSVM<> lsvm;
	UnserializeObject<mlpack::LinearSVM<>>(model, lsvm);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::Row<size_t> classifiedvalues(n);
	lsvm.Classify(dataset, classifiedvalues);
	if (verbose)
		classifiedvalues.print("predicted");

	output.SetChildCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)classifiedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
