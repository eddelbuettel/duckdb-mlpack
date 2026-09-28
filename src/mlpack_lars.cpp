#include <mlpack_lars.hpp>
#include <mlpack/methods/lars/lars.hpp>

namespace duckdb {

void MlpackLARSTrainTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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

	const double lambda1 = params.count("lambda1") > 0 ? std::stod(params["lambda1"]) : 0.0;
	const double lambda2 = params.count("lambda2") > 0 ? std::stod(params["lambda2"]) : 0.0;
	const bool useCholesky = params.count("use_cholesky") > 0 ? (params["use_cholesky"] == "true" ? true : false) : true;
	const bool noIntercept = params.count("no_intercept") > 0 ? (params["no_intercept"] == "true" ? true : false) : true;
	const bool noNormalize = params.count("no_normalize") > 0 ? (params["no_normalize"] == "true" ? true : false) : true;
	if (params.count("silent") > 0)
		silent = (params["silent"] == "true" ? true : false);

	mlpack::LARS<>* lars = new mlpack::LARS<>(useCholesky, lambda1, lambda2);
	lars->FitIntercept(!noIntercept);
	lars->NormalizeData(!noNormalize);
	lars->Train(dataset, labelsvec, true /* transpose */);

	if (verbose)
		std::cout << SerializeObject<mlpack::LARS<>>(*lars) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::LARS<>>(*lars));

	auto n = labelsvec.n_elem;
	arma::rowvec fittedvalues(n);
	lars->Predict(dataset, fittedvalues);
	if (verbose)
		fittedvalues.print("fitted");
	auto rmse = std::sqrt(arma::as_scalar(arma::mean(arma::square(labelsvec - fittedvalues))));
	if (!silent)
		std::cout << "RMSE: " << rmse << std::endl;

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, fittedvalues[i]);
	}

	delete lars;

	resdata.data_returned = true; // mark that we have been called
}

void MlpackLARSPredictTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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

	mlpack::LARS lars;
	UnserializeObject<mlpack::LARS<>>(model, lars);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::rowvec fittedvalues(n);
	lars.Predict(dataset, fittedvalues);
	if (verbose)
		fittedvalues.print("fitted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, fittedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
