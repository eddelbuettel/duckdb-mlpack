#include <mlpack_linear_regression.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

namespace duckdb {

unique_ptr<FunctionData> MlpackLinRegTableBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>(); // 'resdata' for result data i.e. outgoing
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->labels = input.inputs[1].GetValue<std::string>();
	resdata->parameters = input.inputs[2].GetValue<std::string>();
	resdata->model = input.inputs[3].GetValue<std::string>();
	names = {"fitted"};
	return_types = {LogicalType::DOUBLE};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

void MlpackLinRegTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = false;
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
	// Dependent variable i.e. 'labels'
	arma::Row<double> labelsvec = get_armadillo_row<double>(context, resdata.labels);
	if (verbose)
		labelsvec.print("labelsvec");
	std::map<std::string, std::string> params = get_parameters(context, resdata.parameters);

	const double lambda = params.count("lambda") > 0 ? std::stoi(params["lambda"]) : 0.0;
	const bool intercept = params.count("intercept") > 0 ? (params["intercept"] == "true" ? true : false) : true;
	if (verbose) {
		std::cout << "lambda : " << lambda << std::endl;
		std::cout << "intercept : " << (intercept ? "yes" : "no") << std::endl;
	}
	mlpack::LinearRegression lr(dataset, labelsvec, lambda, intercept);

	if (verbose)
		std::cout << SerializeObject<mlpack::LinearRegression<>>(lr) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::LinearRegression<>>(lr));

	auto n = labelsvec.n_elem;
	arma::rowvec fittedvalues(n);
	lr.Predict(dataset, fittedvalues);
	if (verbose)
		fittedvalues.print("fitted");
	auto rmse = std::sqrt(arma::as_scalar(arma::mean(arma::square(labelsvec - fittedvalues))));
	std::cout << "RMSE: " << rmse << std::endl;

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, fittedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

unique_ptr<FunctionData> MlpackLinRegPredTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>(); // 'resdata' for result data i.e. outgoing
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->model = input.inputs[1].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::DOUBLE};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

void MlpackLinRegPredTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = false;
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

	mlpack::LinearRegression lr;
	UnserializeObject<mlpack::LinearRegression<>>(model, lr);

	auto n = dataset.n_cols; // cols not rows because transposed
	arma::rowvec fittedvalues(n);
	lr.Predict(dataset, fittedvalues);
	if (verbose)
		fittedvalues.print("fitted");

	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, fittedvalues[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
