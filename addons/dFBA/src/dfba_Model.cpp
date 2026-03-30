/*
 dFBAintracellular.cpp
 *
 *  Created on: 11 jun. 2019
 *      Author: mponce
 */

#include "dfba_Model.h"
#include "dfba_Reaction.h"  // Include the full definition

/* Default dFBAModel used to initialize the initial cell */
dFBAModel default_dFBAModel;


dFBAModel::dFBAModel()
{
    this->id = "none";
    this->is_initialized = false;
    this->solution.status = "none";
}

dFBAModel::~dFBAModel() {
    for(dFBAReaction* rxn: this->reactions)
        delete rxn;

    for(dFBAMetabolite* met: this->metabolites)
        delete met;

}

dFBAModel::dFBAModel(const dFBAModel& copy) {
    // Copy primitive members
    this->id = copy.id;
    this->is_initialized = copy.is_initialized;

    // Deep copy metabolites
    for (auto original_metabolite : copy.metabolites) {
        this->metabolites.push_back(new dFBAMetabolite(*original_metabolite));
    }

    // Deep copy reactions
    for (auto original_reaction : copy.reactions) {
        this->reactions.push_back(new dFBAReaction(*original_reaction));
    }

    // Copy maps
    this->metaboliteIndexer = copy.metaboliteIndexer;
    this->reactionsIndexer = copy.reactionsIndexer;

    // Copy solution (assuming it can be copied by value)
    this->solution = copy.solution;

    // Re-initialize HiGHS problem for the copy
    if (copy.is_initialized) {
        this->initProblem();
    }

}



dFBAModel& dFBAModel::operator=(const dFBAModel& other) {
    if (this == &other) {
        return *this; // Handle self-assignment
    }

    // Clean up existing resources to prevent memory leaks
    for (auto met : metabolites) {
        delete met;
    }
    metabolites.clear();

    for (auto rxn : reactions) {
        delete rxn;
    }
    reactions.clear();

    // Copy primitive members
    this->id = other.id;
    this->is_initialized = other.is_initialized;

    // Deep copy metabolites
    for (auto original_metabolite : other.metabolites) {
        this->metabolites.push_back(new dFBAMetabolite(*original_metabolite));
    }

    // Deep copy reactions
    for (auto original_reaction : other.reactions) {
        this->reactions.push_back(new dFBAReaction(*original_reaction));
    }

    // Copy other maps
    this->metaboliteIndexer = other.metaboliteIndexer;
    this->reactionsIndexer = other.reactionsIndexer;

    // Copy solution (assuming it can be copied by value)
    this->solution = other.solution;

    // Re-initialize HiGHS problem
    if (other.is_initialized) {
        this->initProblem(); 
    }

    return *this;
}



void dFBAModel::clear(){

    this->id = "none";
    this->is_initialized = false;
    this->reactions.clear();
    this->metabolites.clear();
    this->reactionsIndexer.clear();
    this->metaboliteIndexer.clear();
    this->solution.clear();
    highs.~Highs();
    new (&highs) Highs();

    return;
}

const int dFBAModel::getNumReactions()
{
    return this->reactions.size();
}

const int dFBAModel::getNumMetabolites()
{
    return this->metabolites.size();
}

bool dFBAModel::hasMetabolite(const std::string& id) const
{
    return this->metaboliteIndexer.find(id) != this->metaboliteIndexer.end();
}

dFBAMetabolite* dFBAModel::addMetabolite(const std::string& id)
{
    // Check if metabolite already exists by its ID
    auto it = this->metaboliteIndexer.find(id);
    if (it == this->metaboliteIndexer.end()) {
        // If metabolite does not exist, create it and add it to the model
        dFBAMetabolite* newMet = new dFBAMetabolite(id);
        this->metabolites.push_back(newMet);
        this->metaboliteIndexer[id] = this->metabolites.size() - 1;
        return newMet;
    } else {
        // If metabolite already exists, return the existing one
        return this->metabolites[it->second];
    }
}

dFBAMetabolite* dFBAModel::getMetabolite(const std::string& mId) const {
    auto it = this->metaboliteIndexer.find(mId);
    if (it != this->metaboliteIndexer.end()) {
        int idx = it->second;
        return this->metabolites[idx];
    }
    return nullptr;  // Return nullptr if metabolite doesn't exist
}



const std::vector<dFBAMetabolite*> dFBAModel::getListOfMetabolites() const
{
    return this->metabolites;
}

bool dFBAModel::hasReaction(std::string rId)
{
    std::map<std::string, int>::iterator it;
    it = this->reactionsIndexer.find(rId);

    return it != this->reactionsIndexer.end();
}

dFBAReaction* dFBAModel::getReaction(std::string rId)
{
    if (this->hasReaction(rId))
    {
        int idx = this->reactionsIndexer[rId];
        dFBAReaction* rxn = this->reactions[idx];
        return rxn;
    }
    return nullptr;
}


double dFBAModel::getReactionUpperBound(std::string rId) 
{
    dFBAReaction* rxn = this->getReaction(rId);
    if (rxn)
        return rxn->getUpperBound();
    else {
        std::cerr << "Error: Reaction " << rId << " not found in getReactionUpperBound" << std::endl;
        return 0.0;
    }
}

void dFBAModel::setReactionUpperBound(std::string rId, double upperBound)
{
    dFBAReaction* rxn = this->getReaction(rId);
    if (rxn)
    {
        rxn->setUpperBound(upperBound);
        int colIdx = this->reactionsIndexer[rId];
        this->highs.changeColBounds(colIdx, rxn->getLowerBound(), upperBound);
    }
    else{
        std::cerr << "Reaction with ID " << rId << " not found in the model." << std::endl;
    }
}

double dFBAModel::getReactionLowerBound(std::string rId) 
{
    dFBAReaction* rxn = this->getReaction(rId);
    if (rxn)
        return rxn->getLowerBound();
    else {
        std::cerr << "Error: Reaction " << rId << " not found in getReactionLowerBound" << std::endl;
        return 0.0;
    }
}

void dFBAModel::setReactionLowerBound(std::string rId, double lowerBound)
{
    dFBAReaction* rxn = this->getReaction(rId);
    if (rxn)
    {
        rxn->setLowerBound(lowerBound);
        int colIdx = this->reactionsIndexer[rId];
        this->highs.changeColBounds(colIdx, lowerBound , rxn->getUpperBound());
    }
    else{
        std::cerr << "Reaction with ID " << rId << " not found in the model." << std::endl;
    }

}

void dFBAModel::addReaction(dFBAReaction* rxn)
{
    if (this->is_initialized) {
        std::cerr << "CRITICAL ERROR: Cannot add reactions after model initialization." << std::endl;
        exit(-1);
    }
    if (!this->hasReaction( rxn->getId() ))
    {
        this->reactions.push_back(rxn);
        this->reactionsIndexer[rxn->getId()] = this->reactions.size() - 1;
    }
}

const int dFBAModel::getReactionIndex(std::string rId)
{
    if (this->hasReaction(rId))
        return this->reactionsIndexer[rId];
    else
        return -1;
}

const std::vector<dFBAReaction*> dFBAModel::getListOfReactions() const
{
    return this->reactions;
}

std::vector<dFBAReaction*> dFBAModel::getListOfBoundaryReactions()
{
    std::vector<dFBAReaction*> listOfBoundarys;
    for(dFBAReaction* dFBAReaction: this->reactions)
    {
        if (dFBAReaction->getNumberOfMetabolites() == 1)
        {
            listOfBoundarys.push_back(dFBAReaction);        
        }
    }
    return listOfBoundarys;
}

std::vector<std::string> dFBAModel::getListOfBoundaryReactionIds()
{
    std::vector<std::string> listOfBoundaryIds;
    for(dFBAReaction* dFBAReaction: this->reactions)
    {
        if (dFBAReaction->getNumberOfMetabolites() == 1)
        {
            listOfBoundaryIds.push_back(dFBAReaction->getId());
        }
    }
    return listOfBoundaryIds;
}

void dFBAModel::readSBMLModel(const char* sbmlFileName)
{
    SBMLReader reader;
    SBMLDocument* document = reader.readSBML(sbmlFileName);

    // Check if the document was successfully read
    if (document == nullptr) {
    std::cerr << "Error: SBMLDocument is null for " << sbmlFileName << "\n";
    return;
    }

    // Optional: run consistency checks (can add more warnings)
    // document->checkConsistency();

    const unsigned nFatal = document->getNumErrors(LIBSBML_SEV_FATAL);
    const unsigned nError = document->getNumErrors(LIBSBML_SEV_ERROR);
    const unsigned nWarn  = document->getNumErrors(LIBSBML_SEV_WARNING);

    if (nFatal + nError > 0) {
        std::cerr << "Failed to read SBML: " << sbmlFileName << "\n";
        document->printErrors();  // show why it failed
        delete document;
        return;
    }

    // Non-fatal: show warnings but continue
    if (nWarn) {
        std::cerr << "Validation warning(s): " << nWarn << "\n";
        document->printErrors();
    }

    Model* model = document->getModel();

    // Check if the model was successfully retrieved
    if (model == nullptr) {
        std::cerr << "Error: Model is null in SBML file: " << sbmlFileName << std::endl;
        delete document;
        return;
    }

    ListOfSpecies* listOfSpecies = model->getListOfSpecies();
    ListOfReactions* listOfReactions = model->getListOfReactions();
    ListOfParameters* listOfParameters = model->getListOfParameters();

    this->id = model->getId();

    for (unsigned int i = 0; i < model->getNumSpecies(); i++)
    {
        Species* species = listOfSpecies->get(i);
        // Skipping boundary dFBAMetabolites
        if ( species->getBoundaryCondition() )
            continue;

        dFBAMetabolite* metabolite = new dFBAMetabolite(species->getId());
        metabolite->setName(species->getName());
        this->addMetabolite(metabolite->getId());
    }

    for(unsigned int i = 0; i < model->getNumReactions(); i++)
    {
        Reaction* sbml_reaction = listOfReactions->get(i);

        dFBAReaction* reaction = new dFBAReaction(sbml_reaction->getId());
        // std::cout << "Adding reaction with ID: " << reaction->getId() << std::endl;
        reaction->setName(sbml_reaction->getName());

        FbcReactionPlugin* rxnFbc = static_cast<FbcReactionPlugin*> (sbml_reaction->getPlugin("fbc"));
        if ( rxnFbc )
        {
            // Getting dFBAReaction's upper and lower bounds
            std::string lbId = rxnFbc->getLowerFluxBound();
            Parameter* lbParam = listOfParameters->get(lbId);
            if (lbParam) {
                double lb = lbParam->getValue();
                reaction->setLowerBound(lb);
            }
            else {
                std::cout << "WARNING: Lower bound parameter " << lbId << " not found for reaction " << sbml_reaction->getId() << std::endl;
            }

            std::string ubId = rxnFbc->getUpperFluxBound();
            Parameter* ubParam = listOfParameters->get(ubId);
            if (ubParam) {
                double ub = ubParam->getValue();
                reaction->setUpperBound(ub);
            }
            else {
                std::cout << "WARNING: Upper bound parameter " << ubId << " not found for reaction " << sbml_reaction->getId() << std::endl;
            }
        }
        else {
            std::cout << "WARNING: No FBC plugin found for reaction " << sbml_reaction->getId() << std::endl;
        }
        int numReactans = sbml_reaction->getNumReactants();
        for(int j = 0; j < numReactans; j++)
        {
            SpeciesReference* sbml_species = sbml_reaction->getReactant(j);
            double stoich_coef = -1. * sbml_species->getStoichiometry();

            if ( !this->hasMetabolite(sbml_species->getSpecies()) )
            {
                dFBAMetabolite* metabolite = new dFBAMetabolite(sbml_species->getSpecies());
                metabolite->setName(sbml_species->getName());
                this->addMetabolite(metabolite->getId());
            }
            
            const dFBAMetabolite* metabolite = this->getMetabolite(sbml_species->getSpecies());
            if (metabolite != nullptr)
                reaction->addMetabolite(metabolite->getId(), stoich_coef);
            else
                std::cout << "ERROR: dFBAMetabolite " << sbml_species->getSpecies() << " not found" << std::endl;
        }

        int numProducts = sbml_reaction->getNumProducts();
        for(int j = 0; j < numProducts; j++)
        {
            SpeciesReference* sbml_species = sbml_reaction->getProduct(j);
            double stoich_coef = sbml_species->getStoichiometry();

            if ( !this->hasMetabolite(sbml_species->getSpecies()) )
            {
                dFBAMetabolite* metabolite = new dFBAMetabolite(sbml_species->getSpecies());
                metabolite->setName(sbml_species->getName());
                this->addMetabolite(metabolite->getId());
            }
            const dFBAMetabolite* metabolite = this->getMetabolite(sbml_species->getSpecies());
            if (metabolite != nullptr)
                reaction->addMetabolite(metabolite->getId(), stoich_coef);
            else
                std::cout << "ERROR: dFBAMetabolite " << sbml_species->getSpecies() << " not found" << std::endl;
        }
        this->addReaction(reaction);
    }

    // The following code is intended to extract the objective function from the sbml using
    // the FbcdFBAModelPlugin; then the coefficients are assigned to the corresponding dFBAReactions
    FbcModelPlugin* mplugin = static_cast<FbcModelPlugin*>(model->getPlugin("fbc"));
    if (mplugin) {
        ListOfObjectives* listOfObjectives = mplugin->getListOfObjectives();
        Objective* objective = mplugin->getObjective(listOfObjectives->getActiveObjective());
        ListOfFluxObjectives* listOfFluxObjectives = objective->getListOfFluxObjectives();

        for (unsigned int i = 0; i < listOfFluxObjectives->getNumFluxObjectives(); i++)
        {
            FluxObjective* fluxObjective = listOfFluxObjectives->get(i);
            std::string rId = fluxObjective->getReaction();
            double objectiveCoefficient = fluxObjective->getCoefficient();
            dFBAReaction* dFBAReaction = this->getReaction(rId);
            if (dFBAReaction) {
                dFBAReaction->setObjectiveCoefficient(objectiveCoefficient);
            } else {
                std::cerr << "ERROR: dFBAReaction " << rId << " not found" << std::endl;
            }
        }
    }

    delete document;
}

void dFBAModel::initProblem() {
    int n_rows = this->getNumMetabolites();
    int n_cols = this->getNumReactions();

    // Silence HiGHS output
    highs.setOptionValue("output_flag", false);

    std::cout << "Initializing LP problem with " << n_rows << " constraints and "
              << n_cols << " variables." << std::endl;

    // Prepare LP data vectors.
    std::vector<double> row_lower(n_rows, 0.0);  // All row lower bounds set to 0.
    std::vector<double> row_upper(n_rows, 0.0);    // All row upper bounds set to 0.
    std::vector<double> col_lower(n_cols);
    std::vector<double> col_upper(n_cols);
    std::vector<double> col_cost(n_cols);

    // To process reactions in order, create a vector indexed by column index.
    std::vector<dFBAReaction*> reactions_by_index(n_cols, nullptr);
    for (dFBAReaction* rxn : this->reactions) {
      int col_idx = this->reactionsIndexer[rxn->getId()];
      reactions_by_index[col_idx] = rxn;
    }
  
    // Prepare the packed column-wise matrix data.
    std::vector<int> a_start(n_cols + 1, 0); // a_start[0] must be zero.
    std::vector<int> a_index;      // Row indices for nonzeros.
    std::vector<double> a_value;   // Nonzero values.
  
    // Loop over each reaction (column).
    for (int j = 0; j < n_cols; j++) {
      dFBAReaction* rxn = reactions_by_index[j];
      // Set column bounds and objective coefficients.
      col_lower[j] = rxn->getLowerBound();
      col_upper[j] = rxn->getUpperBound();
      col_cost[j]  = rxn->getObjectiveCoefficient();
  
      // Record starting index for column j.
      a_start[j] = a_index.size();
      // For each metabolite in the reaction, add the nonzero entry.
      const std::map<std::string, double>& mets = rxn->getMetabolites();
      for (auto it = mets.begin(); it != mets.end(); ++it) {
        auto idx_it = this->metaboliteIndexer.find(it->first);
        if (idx_it != this->metaboliteIndexer.end()) {
          int row_idx = idx_it->second;
          a_index.push_back(row_idx);
          a_value.push_back(it->second);
        }
      }
    }
    // Final entry: total number of nonzeros.
    a_start[n_cols] = a_index.size();

    // Build the HighsModel.
    HighsModel model;
    model.lp_.num_col_ = n_cols;
    model.lp_.num_row_ = n_rows;
    model.lp_.sense_ = ObjSense::kMaximize;
    model.lp_.offset_ = 0.0;
    model.lp_.col_cost_ = col_cost;
    model.lp_.col_lower_ = col_lower;
    model.lp_.col_upper_ = col_upper;
    model.lp_.row_lower_ = row_lower;
    model.lp_.row_upper_ = row_upper;

    // Set the constraint matrix data.
    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = a_start;
    model.lp_.a_matrix_.index_ = a_index;
    model.lp_.a_matrix_.value_ = a_value;

    // Pass the model to HiGHS.
    HighsStatus status = highs.passModel(model);
    if (status != HighsStatus::kOk) {
      std::cerr << "Error passing LP model to HiGHS" << std::endl;
    }

    // Set solver options
    highs.setOptionValue("primal_feasibility_tolerance", 1e-7);
    highs.setOptionValue("dual_feasibility_tolerance", 1e-7);
    highs.setOptionValue("simplex_iteration_limit", 10000);

    // Perform initial solve
    highs.run();

    this->is_initialized = true;
  }
  


void dFBAModel::initModel(const char* sbmlFileName)
{
    this->readSBMLModel(sbmlFileName);
    std::cout << "SBML model correctly loaded: " << sbmlFileName << std::endl;
    this->initProblem();
}

void dFBAModel::writeProblem(const char *filename)
{
    highs.writeModel(filename); // if such a function exists
}

dFBASolution dFBAModel::optimize()
{
    // Solve the model using HiGHS
    HighsStatus run_status = highs.run();

    solution.fluxes.clear();

    // Get model status
    HighsModelStatus model_status = highs.getModelStatus();
    std::string status;
    bool isOptimal = (model_status == HighsModelStatus::kOptimal);

    if (isOptimal) {
        status = "optimal";
    } else if (model_status == HighsModelStatus::kInfeasible) {
        status = "infeasible";
    } else {
        status = "unknown";
        std::cerr << "Solver returned status: " << highs.modelStatusToString(model_status) << std::endl;
    }

    if (isOptimal)
    {
        const HighsSolution& sol = highs.getSolution();
        double fopt = highs.getInfo().objective_function_value;

        for (dFBAReaction* reaction : this->reactions)
        {
            int column_idx = this->reactionsIndexer[reaction->getId()];
            double flux = sol.col_value[column_idx];
            solution.fluxes[reaction->getId()] = flux;
            reaction->setFluxValue(flux);
        }

        solution.objective_value = fopt;
        solution.status = status;
    }
    else if (status == "infeasible")
    {
        solution.status = status;

        for (auto &reaction : this->reactions)
        {
            reaction->setFluxValue(0.0);
        }
        this->solution.objective_value = 0.0;
    }
    else
    {
        solution.status = status;

        std::cerr << "ERROR: FBA optimization failed with unknown status!\n";
        std::cerr << "Reaction bounds at failure:\n";
        for (auto &reaction : this->reactions)
        {
            double lb = reaction->getLowerBound();
            double ub = reaction->getUpperBound();
            std::cerr << reaction->getId() << ": [" << lb << ", " << ub << "]\n";
            reaction->setFluxValue(0.0);
        }
    }

    return solution;
}


bool dFBAModel::getSolutionStatus()
{
    if (!this->is_initialized) return false;
    HighsModelStatus status = highs.getModelStatus();
    return (status == HighsModelStatus::kOptimal);
}

double dFBAModel::getObjectiveValue()
{
    assert(this->is_initialized);
    HighsModelStatus modelStatus = highs.getModelStatus();
    if (modelStatus == HighsModelStatus::kOptimal) {
      return highs.getInfo().objective_function_value;
    } else {
      std::cout << "WARNING: Primal infeasible or model not optimal" << std::endl;
      return 0;
    }
}

bool dFBAModel::isInitialized(){
    return this->is_initialized;
}