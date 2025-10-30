#include <mlpack_logistic_regression.hpp>
#include <mlpack/methods/logistic_regression.hpp>

namespace duckdb {

unique_ptr<FunctionData> MlpackLogisticRegTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>(); // 'resdata' for result data i.e. outgoing
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->labels = input.inputs[1].GetValue<std::string>();
	resdata->parameters = input.inputs[2].GetValue<std::string>();
	resdata->model = input.inputs[3].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::INTEGER};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

void MlpackLogisticRegTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = false;
	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetCardinality(0);
		return;
	}

	// Explanatory variables i.e. 'features'
	arma::mat dataset = get_armadillo_matrix_transposed<double>(context, resdata.features);
	// Dependent variable i.e. 'labels'
	arma::Row<size_t> labelsvec = get_armadillo_row<size_t>(context, resdata.labels);
	std::map<std::string, std::string> params = get_parameters(context, resdata.parameters);

	const double lambda = params.count("lambda") > 0 ? std::stod(params["lambda"]) : 0.0;
	const bool silent = params.count("silent") > 0 ? (params["silent"] == "true" ? true : false) : false;

	mlpack::LogisticRegression lr(dataset, labelsvec, lambda);

	if (verbose)
		std::cout << SerializeObject<mlpack::LogisticRegression<>>(lr) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::LogisticRegression<>>(lr));

	auto avec = lr.Parameters().t();
	if (verbose)
		std::cout << "Coefficients:" << serialize_vector(avec) << std::endl;
	store_vector(context, resdata.model, "coefficients", serialize_vector(avec));

	arma::Row<size_t> predictions;
	arma::mat probabilities;
	lr.Classify(dataset, predictions, probabilities);
	store_vector(context, resdata.model, "predictions", serialize_vector(arma::conv_to<arma::vec>::from(predictions)));
	store_vector(context, resdata.model, "probabilities_0",
	             serialize_vector(arma::conv_to<arma::vec>::from(probabilities.row(0))));
	store_vector(context, resdata.model, "probabilities_1",
	             serialize_vector(arma::conv_to<arma::vec>::from(probabilities.row(1))));

	if (verbose) {
		predictions.print("predictions");
		probabilities.print("probabilties");
	}
	size_t countError = arma::accu(labelsvec != predictions);
	if (!silent)
		std::cout << "Misclassified: " << countError << std::endl;

	auto n = labelsvec.n_elem;
	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)predictions[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

unique_ptr<FunctionData> MlpackLogisticRegPredTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {
	auto resdata = make_uniq<MlpackModelData>(); // 'resdata' for result data i.e. outgoing
	resdata->features = input.inputs[0].GetValue<std::string>();
	resdata->model = input.inputs[1].GetValue<std::string>();
	names = {"predicted"};
	return_types = {LogicalType::INTEGER};
	resdata->return_types = return_types;
	resdata->names = names;
	return std::move(resdata);
}

void MlpackLogisticRegPredTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	bool verbose = false;
	auto &resdata = const_cast<MlpackModelData &>(data_p.bind_data->Cast<MlpackModelData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetCardinality(0);
		return;
	}

	// Explanatory variables i.e. 'features'
	arma::mat dataset = get_armadillo_matrix_transposed<double>(context, resdata.features);
	if (verbose)
		dataset.print("dataset");

	auto model = retrieve_model(context, resdata.model);
	if (verbose)
		std::cout << model << std::endl;

	mlpack::LogisticRegression lr;
	UnserializeObject<mlpack::LogisticRegression<>>(model, lr);

	arma::Row<size_t> predictions;
	arma::mat probabilities;
	lr.Classify(dataset, predictions, probabilities);
	if (verbose) {
		predictions.print("predictions");
		probabilities.print("probabilties");
	}

	auto n = predictions.n_elem;
	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int32_t)predictions[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

} // namespace duckdb
