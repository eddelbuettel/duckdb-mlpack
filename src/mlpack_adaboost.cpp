
#include "mlpack_adaboost.hpp"

namespace duckdb {

static bool verbose = false;

// mlpack adaboost accessor

unique_ptr<FunctionData> MlpackAdaboostTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	if (verbose)
		std::cout << "MlpackAdaboostTableBind\n";
	auto resdata = make_uniq<MlpackModelData>();
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

void MlpackAdaboostTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

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
	// Dependent variable i.e. 'labels'
	arma::Row<size_t> labelsvec = get_armadillo_row<size_t>(context, resdata.labels);
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

	double ztProduct = a.Train(dataset, labelsvec, numClasses, iterations, tolerance, perceptronIter);

	if (verbose)
		std::cout << SerializeObject<mlpack::AdaBoost<PerceptronType, arma::mat>>(a) << std::endl;
	store_model(context, resdata.model, SerializeObject<mlpack::AdaBoost<PerceptronType, arma::mat>>(a));

	arma::Row<size_t> predictedLabels;
	a.Classify(dataset, predictedLabels);
	if (verbose)
		predictedLabels.print("predicted");
	size_t countError = arma::accu(labelsvec != predictedLabels);
	std::cout << "Misclassified: " << countError << std::endl;

	auto n = predictedLabels.n_elem;
	output.SetCardinality(n);
	for (idx_t i = 0; i < n; i++) {
		output.data[0].SetValue(i, (int)predictedLabels[i]);
	}

	resdata.data_returned = true; // mark that we have been called
}

unique_ptr<FunctionData> MlpackAdaboostPredTableBind(ClientContext &context, TableFunctionBindInput &input,
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

void MlpackAdaboostPredTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
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
