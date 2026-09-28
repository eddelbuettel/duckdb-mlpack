#include <mlpack_bayesian_linear_regression.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression.hpp>

namespace duckdb {

void MlpackBayesianLinearRegressionTrainTableFunction(ClientContext &context, TableFunctionInput &data_p,
                                                      DataChunk &output) {
	bool verbose = get_setting<bool>(context, "mlpack_verbose");
	bool silent = get_setting<bool>(context, "mlpack_silent");

	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called before, return nothing
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
	// Dependent variable i.e. 'labels'
	arma::Row<double> labelsvec = get_armadillo_row<double>(context, resdata.labels);
	if (verbose)
		labelsvec.print("labelsvec");
	std::map<std::string, std::string> params = get_parameters(context, resdata.parameters);

	const bool center = params.count("center") > 0 ? (params["center"] == "true" ? true : false) : true;
	const bool scale = params.count("scale") > 0 ? (params["scale"] == "true" ? true : false) : true;
	// also: max_iteration and tolerance
	if (params.count("silent") > 0)
		silent = (params["silent"] == "true" ? true : false);
	mlpack::BayesianLinearRegression blr(dataset, labelsvec, center, scale);

	if (verbose)
		std::cout << SerializeObject<mlpack::BayesianLinearRegression<>>(blr) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::BayesianLinearRegression<>>(blr));

	auto avec = blr.Omega(); // 'solution' vector
	if (verbose)
		std::cout << "Coefficients:" << serialize_vector(avec) << std::endl;
	store_vector(context, resdata.model, "coefficients", serialize_vector(avec));

	auto n = labelsvec.n_elem;
	arma::rowvec fittedvalues(n);
	blr.Predict(dataset, fittedvalues);
	if (verbose)
		fittedvalues.print("fitted");
	auto rmse = std::sqrt(arma::as_scalar(arma::mean(arma::square(labelsvec - fittedvalues))));
	if (!silent)
		std::cout << "RMSE: " << rmse << std::endl;

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, fittedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

void MlpackBayesianLinearRegressionPredictTableFunction(ClientContext &context, TableFunctionInput &data_p,
                                                        DataChunk &output) {
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

	mlpack::BayesianLinearRegression blr;
	UnserializeObject<mlpack::BayesianLinearRegression<>>(model, blr);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::rowvec fittedvalues(n);
	blr.Predict(dataset, fittedvalues);
	if (verbose)
		fittedvalues.print("fitted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, fittedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
