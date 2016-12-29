#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <strstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

bool DEBUG = false;
#define MAX_LEN 256
using namespace std;
using namespace dynet;
namespace po = boost::program_options;

float pdrop = 0.5;
float unk_prob = 0.2;
float margin = 1;

//word
unsigned WORD_DIM = 50;
unsigned WORD_feature_n = 18;
unsigned POSTAG_DIM = 50;
unsigned POSTAG_feature_n = 18;
unsigned LABEL_DIM = 50;
unsigned LABEL_feature_n = 12;
unsigned ACTION_DIM = 50;

unsigned LAYERS = 1;

unsigned HIDDEN_DIM = 200;
unsigned LSTM_HIDDEN_DIM = 100;

unsigned POSTAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned LABEL_SIZE = 0;
unsigned ACTION_SIZE = 0;

dynet::Dict wd;
dynet::Dict td;
dynet::Dict ld;
dynet::Dict ad;
unsigned kUNK;
unsigned lROOT;
unsigned tROOT;
unsigned none_label;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data", po::value<string>(), "Test corpus")
        ("pdrop", po::value<float>()->default_value(0.5), "dropout probabilty")
	("unk_prob,u", po::value<float>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("margin", po::value<float>()->default_value(1.0), "margin")
	("eta", po::value<float>()->default_value(0.1), "eta")
	("eta_decay", po::value<float>()->default_value(0.1), "eta_decay")
	("model,m", po::value<string>(), "Load saved model from this file")
        ("word_dim", po::value<unsigned>()->default_value(50), "word embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(200), "hidden dimension")
        ("pos_dim", po::value<unsigned>()->default_value(50), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(50), "relation dimension")
        ("act_dim", po::value<unsigned>()->default_value(50), "action dimension")
	("lstm_hidden_dim", po::value<unsigned>()->default_value(100), "lstm hidden dimension")
	("layers", po::value<unsigned>()->default_value(1), "lstm layers")
	("train,t", "Should training be run?")
        ("batch_size", po::value<unsigned>()->default_value(20), "batch size")
        ("report_i", po::value<unsigned>()->default_value(50), "report i")
        ("dev_report_i", po::value<unsigned>()->default_value(25), "dev report i")
        ("train_methods", po::value<unsigned>()->default_value(0), "0 for simple, 1 for mon, 2 for adagrad, 3 for adam")
	("debug", "Debug to output trace")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0 || conf->count("dev_data") == 0 || conf->count("test_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    cerr << "Also specify --dev_data (-d)\n";
    exit(1);
  }
}



class StateItem{
public:
	int dep_l1[MAX_LEN];
	int dep_l2[MAX_LEN];
	unsigned arc_l1[MAX_LEN];
	unsigned arc_l2[MAX_LEN];
	int dep_r1[MAX_LEN];
	int dep_r2[MAX_LEN];
	unsigned arc_r1[MAX_LEN];
	unsigned arc_r2[MAX_LEN];

	int dep[MAX_LEN];
	unsigned label[MAX_LEN];

	vector<int> stack;
	vector<int> buffer;
	StateItem(){
		clear();
	};
	StateItem(const StateItem& stateitem){
		memcpy(dep_l1, stateitem.dep_l1, sizeof(int)*MAX_LEN);
		memcpy(dep_l2, stateitem.dep_l2, sizeof(int)*MAX_LEN);
		memcpy(dep_r1, stateitem.dep_r1, sizeof(int)*MAX_LEN);
		memcpy(dep_r2, stateitem.dep_r2, sizeof(int)*MAX_LEN);
		memcpy(arc_l1, stateitem.arc_l1, sizeof(unsigned)*MAX_LEN);
		memcpy(arc_l2, stateitem.arc_l2, sizeof(unsigned)*MAX_LEN);
		memcpy(arc_r1, stateitem.arc_r1, sizeof(unsigned)*MAX_LEN);
		memcpy(arc_r2, stateitem.arc_r2, sizeof(unsigned)*MAX_LEN);
		memcpy(dep, stateitem.dep, sizeof(int)*MAX_LEN);
		memcpy(label, stateitem.label, sizeof(unsigned)*MAX_LEN);
		stack = stateitem.stack;
		buffer = stateitem.buffer;
	};
	~StateItem(){};
	void clear(){
		memset(dep_l1, -1, sizeof(int)*MAX_LEN);
                memset(dep_l2, -1, sizeof(int)*MAX_LEN);
                memset(dep_r1, -1, sizeof(int)*MAX_LEN);
                memset(dep_r2, -1, sizeof(int)*MAX_LEN);
		memset(dep, -1, sizeof(int)*MAX_LEN);
		for(unsigned i = 0; i < MAX_LEN; ++i){
			arc_l1[i] = none_label;
			arc_l2[i] = none_label;
			arc_r1[i] = none_label;
			arc_r2[i] = none_label;
			label[i] = none_label;
		}
		stack.clear();
                buffer.clear();
	}
	void init(const vector<unsigned>& words, const vector<unsigned>& postags){
		assert(words.size() == postags.size());
		for(int i = words.size()-1; i >=0; i --){
			buffer.push_back(i);
		}
		
	}
	void shows(){
		std::cerr<<"dep-arc l1 "<<std::endl;
		for(int i = 0; i < 20; i ++){
			std::cerr<<dep_l1[i]<<"-";
			if(arc_l1[i] == none_label) std::cerr <<"NONE ";
			else std::cerr << ld.convert(arc_l1[i])<<" ";
		}
		std::cerr<<std::endl;
	
		std::cerr<<"dep-arc l2 "<<std::endl;
                for(int i = 0; i < 20; i ++){
                        std::cerr<<dep_l2[i]<<"-";
                        if(arc_l2[i] == none_label) std::cerr <<"NONE ";
                        else std::cerr << ld.convert(arc_l2[i])<<" ";
                }
                std::cerr<<std::endl;

		std::cerr<<"dep-arc r1 "<<std::endl;
                for(int i = 0; i < 20; i ++){
                        std::cerr<<dep_r1[i]<<"-";
                        if(arc_r1[i] == none_label) std::cerr <<"NONE ";
                        else std::cerr << ld.convert(arc_r1[i])<<" ";
                }
                std::cerr<<std::endl;

		std::cerr<<"dep-arc r2 "<<std::endl;
                for(int i = 0; i < 20; i ++){
                        std::cerr<<dep_r2[i]<<"-";
                        if(arc_r2[i] == none_label) std::cerr <<"NONE ";
                        else std::cerr << ld.convert(arc_r2[i])<<" ";
                }
                std::cerr<<std::endl;

		std::cerr<<"dep "<<std::endl;
                for(int i = 0; i < 20; i ++){
                        std::cerr<<dep[i]<<" ";
                }
                std::cerr<<std::endl;

		std::cerr<<"label "<<std::endl;
                for(int i = 0; i < 20; i ++){
                        std::cerr<<label[i]<<" ";
                }
                std::cerr<<std::endl;

		std::cerr<<"stack"<<std::endl;
		for(unsigned i = 0; i < stack.size(); i ++){
			std::cerr<<stack[i]<<" ";
		}
		std::cerr<<std::endl;

		std::cerr<<"buffer"<<std::endl;
		for(unsigned i = 0; i < buffer.size(); i ++){
			std::cerr<<buffer[i]<<" ";
		}
		std::cerr<<std::endl;
	
	}
	void transit(const string& act, unsigned label){
		if(act == "REDUCE_LEFT"){
			int s1 = stack[stack.size()-1];
			int s2 = stack[stack.size()-2];
			dep[s2] = s1;
			this->label[s2] = label;
			if(dep_l1[s1] == -1){
				dep_l1[s1] = s2;
				arc_l1[s1] = label;
			}
			else if(dep_l1[s1] > s2){
				dep_l2[s1] = dep_l1[s1];
				arc_l2[s1] = arc_l1[s1];
				dep_l1[s1] = s2;
				arc_l1[s1] = label;
			}
			else if(dep_l2[s1] == -1 || dep_l2[s1] > s2){
				dep_l2[s1] = s2;
				arc_l2[s1] = label;
			}
			stack.pop_back();
			stack.pop_back();
			stack.push_back(s1);
		}
		else if(act == "REDUCE_RIGHT"){
			int s1 = stack[stack.size()-1];
                        int s2 = stack[stack.size()-2];
			dep[s1] = s2;
			this->label[s1] = label;
			if(dep_r1[s2] == -1){
				dep_r1[s2] = s1;
                                arc_r1[s2] = label;
			}
			else if(dep_r1[s2] < s1){
				dep_r2[s2] = dep_r1[s2];
				arc_r2[s2] = arc_r1[s2];
				dep_r1[s2] = s1;
				arc_r1[s2] = label;
			}
			else if(dep_r2[s2] == -1 || dep_r2[s2] < s1){
				dep_r2[s2] = s1;
				arc_r2[s2] = label;
			}
			stack.pop_back();
		}
		else if(act == "SHIFT"){
			stack.push_back(buffer.back());
			buffer.pop_back();
		}
		else{
			std::cerr<<"action error:"<<act<<std::endl;
			exit(1);
		}
	}
};

bool valid(const string& word){
        //,?!:;()
        if(word == "," || word == "?" || word == "!" || word == ":" || word == ";")
                return false;
        if(word == "(" || word == ")" || word == "-LRB-" || word == "-RRB-")
                return false;
        //^[.]+$|^[`]+$|^[']+
        if(word == "." || word == ".." || word == "...")
                return false;
        if(word == "`" || word == "``" || word == "```")
                return false;
        if(word == "'" || word == "''" || word == "'''")
                return false;
        return true;
}

void normalize_digital_lower(string& line){
  for(unsigned i = 0; i < line.size(); i ++){
    if(line[i] >= '0' && line[i] <= '9'){
      line[i] = '0';
    }
    else if(line[i] >= 'A' && line[i] <= 'Z'){
      line[i] = line[i] - 'A' + 'a';
    }
  }
}


class Instance{
public:
        vector<unsigned> words;
	vector<unsigned> postags;
	vector<int> deps;
	vector<unsigned> labels;
	vector<unsigned> actions;


/*        Instance(const vector<unsigned>& _words,
		 const vector<unsigned>& _postags,
		 const vector<int>& _deps,
		 const vector<unsigned>& _labels){
                assert(_words.size() == _postags.size() && _words.size() == _deps.size() && _words.size() == _labels.size());
		words = _words;
                postags = _postags;
		deps = _deps;
		labels = _labels;
	};
*/
	Instance(){clear();};
        ~Instance(){};
	void clear(){
		words.clear();
		postags.clear();
		deps.clear();
		labels.clear();
		actions.clear();
	}	
	friend ostream& operator << (ostream& out, Instance& instance){
		for(unsigned i = 0; i < instance.words.size(); i ++){
			    out << wd.convert(instance.words[i]) << "\t"
			    << td.convert(instance.postags[i]) << "\t"
			    << instance.deps[i] << "\t"
			    << ld.convert(instance.labels[i]) << std::endl; 
		}
		out<<"Action"<<std::endl;
		for(unsigned i = 0; i < instance.actions.size(); i ++){
			out << ad.convert(instance.actions[i]);
			out << std::endl;
		}
		return out;
	}
	unsigned size(){ return words.size();}
	void load(const string& line){
                istringstream in(line);
                string word;
                string label;
                int dep;
                while(in>>word) {
                        if(word == "|||") break;
	
			words.push_back(wd.convert(word));
			
                        in >> word;
                        postags.push_back(td.convert(word));

                        in >> dep;
                        deps.push_back(dep);

                        in >> word;
                        if(dep == -1){
                                labels.push_back(ld.convert("ROOT"));
                        }
                        else if((int)labels.size() > dep){
                                labels.push_back(ld.convert("REDUCE_RIGHT "+word));
                        }
                        else{
                                labels.push_back(ld.convert("REDUCE_LEFT "+word));
                        }
                }
                while(in>>word){
                        in>>label;
                        actions.push_back(ad.convert(word+" "+label));
                }

        }

};


struct RNNLanguageModel {
  LookupParameter p_w;
  LookupParameter p_t;
  LookupParameter p_l;
  LookupParameter p_a;

  Parameter p_w_none;
  Parameter p_t_none;
  Parameter p_l_none;

  Parameter p_w2h;
  Parameter p_t2h;
  Parameter p_l2h;
  Parameter p_bias;

  Parameter p_W;
  Parameter p_Sbias;

  LSTMBuilder seq; 
  explicit RNNLanguageModel(Model& model):
	seq(LAYERS, HIDDEN_DIM+ACTION_DIM, LSTM_HIDDEN_DIM,&model){

    p_w = model.add_lookup_parameters(VOCAB_SIZE, {WORD_DIM}, ParameterInitUniform(0.01));
    p_w_none = model.add_parameters({WORD_DIM}, ParameterInitUniform(0.01));
    p_t = model.add_lookup_parameters(POSTAG_SIZE,{POSTAG_DIM}, ParameterInitUniform(0.01));
    p_t_none = model.add_parameters({POSTAG_DIM}, ParameterInitUniform(0.01));
    p_l = model.add_lookup_parameters(LABEL_SIZE, {LABEL_DIM}, ParameterInitUniform(0.01));
    p_l_none = model.add_parameters({LABEL_DIM}, ParameterInitUniform(0.01));
    p_a = model.add_lookup_parameters(ACTION_SIZE, {ACTION_DIM}, ParameterInitUniform(0.01));    

    p_w2h = model.add_parameters({HIDDEN_DIM,WORD_DIM*WORD_feature_n});
    p_t2h = model.add_parameters({HIDDEN_DIM,POSTAG_DIM*POSTAG_feature_n});
    p_l2h = model.add_parameters({HIDDEN_DIM,LABEL_DIM*LABEL_feature_n});
    p_bias = model.add_parameters({HIDDEN_DIM});

    p_W = model.add_parameters({1, LSTM_HIDDEN_DIM});
    p_Sbias = model.add_parameters({1});
  
  }

  // return Expression of total loss
  Expression BuildTaggingGraph(const Instance& inst, ComputationGraph& cg, double* cor = 0, double* predict = 0, unsigned* overall = 0, bool train=true) {
    const unsigned slen = inst.words.size();
    const vector<unsigned>& words = inst.words;
    const vector<unsigned>& postags = inst.postags;
    const vector<unsigned>& labels = inst.labels;
    const vector<unsigned>& actions = inst.actions;
    const vector<int>& deps = inst.deps;
    
    Expression i_w2h = parameter(cg, p_w2h);
    Expression i_t2h = parameter(cg, p_t2h);
    Expression i_l2h = parameter(cg, p_l2h);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_W = parameter(cg, p_W);
    Expression i_Sbias = parameter(cg, p_Sbias);

    StateItem stateitem;
    stateitem.init(inst.words,inst.postags);

    seq.new_graph(cg);
    seq.start_new_sequence();

    assert(slen*2-1 == actions.size());
    vector<Expression> errs;
    vector<Expression> i_words(slen);
    vector<Expression> i_postags(slen);
    
    for(unsigned i = 0; i < words.size(); i ++){
        i_words[i] = lookup(cg, p_w, words[i]);
        if(train) i_words[i] = dropout(i_words[i], pdrop);
	i_postags[i] = lookup(cg, p_t, postags[i]);
        if(train) i_postags[i] = dropout(i_postags[i], pdrop);
    }
    vector<Expression> i_labels(ld.size());
    for(unsigned i = 0; i < ld.size(); i ++){
        i_labels[i] = lookup(cg, p_l, i);
	if(train) i_labels[i] = dropout(i_labels[i], pdrop);
    }
    vector<Expression> i_actions(ad.size());
    for(unsigned i = 0; i < ad.size(); i ++){
  	i_actions[i] = lookup(cg, p_a, i);
	if(train) i_actions[i] = dropout(i_actions[i], pdrop);
    }
    Expression i_w_none = parameter(cg, p_w_none);
    if(train) i_w_none = dropout(i_w_none, pdrop);
    Expression i_t_none = parameter(cg, p_t_none);
    if(train) i_t_none = dropout(i_t_none, pdrop);
    Expression i_l_none = parameter(cg, p_l_none);
    if(train) i_l_none = dropout(i_l_none, pdrop);

    int best_prev = -1;
    int seq_len = 0;
    for(unsigned i = 0; i < slen * 2 -1; ++i){
if(DEBUG){
	std::cerr<<"============= "<<i<<std::endl;
	std::cerr<<actions[i];
	std::cerr<<std::endl;
	stateitem.shows();
}
	vector<Expression> fixed_features;
	vector<Expression> word_features;
	vector<Expression> postag_features;
	vector<Expression> label_features;
	//s1 s2 s3
	int stacksize = stateitem.stack.size();
	int s1 = stacksize > 0 ? stateitem.stack[stacksize-1] : -1;
	int s2 = stacksize > 1 ? stateitem.stack[stacksize-2] : -1;
	int s3 = stacksize > 2 ? stateitem.stack[stacksize-3] : -1;
if(DEBUG)       std::cerr<<"s1:"<<s1<<" s2:"<<s2<<" s3:"<<s3<<std::endl;
        if(s1 != -1) word_features.push_back(i_words[s1]); else word_features.push_back(i_w_none);
        if(s2 != -1) word_features.push_back(i_words[s2]); else word_features.push_back(i_w_none);
        if(s3 != -1) word_features.push_back(i_words[s3]); else word_features.push_back(i_w_none);
        if(s1 != -1) postag_features.push_back(i_postags[s1]); else postag_features.push_back(i_t_none);
        if(s2 != -1) postag_features.push_back(i_postags[s2]); else postag_features.push_back(i_t_none);
        if(s3 != -1) postag_features.push_back(i_postags[s3]); else postag_features.push_back(i_t_none);
        //b1 b2 b3
        int buffersize = stateitem.buffer.size();
        int b1 = buffersize > 0 ? stateitem.buffer[buffersize-1] : -1;
        int b2 = buffersize > 1 ? stateitem.buffer[buffersize-2] : -1;
        int b3 = buffersize > 2 ? stateitem.buffer[buffersize-3] : -1;
if(DEBUG)       std::cerr<<"b1:"<<b1<<" b2:"<<b2<<" b3:"<<b3<<std::endl;
        if(b1 != -1) word_features.push_back(i_words[b1]); else word_features.push_back(i_w_none);
        if(b2 != -1) word_features.push_back(i_words[b2]); else word_features.push_back(i_w_none);
        if(b3 != -1) word_features.push_back(i_words[b3]); else word_features.push_back(i_w_none);
        if(b1 != -1) postag_features.push_back(i_postags[b1]); else postag_features.push_back(i_t_none);
        if(b2 != -1) postag_features.push_back(i_postags[b2]); else postag_features.push_back(i_t_none);
        if(b3 != -1) postag_features.push_back(i_postags[b3]); else postag_features.push_back(i_t_none);
	//lc1(s1), rc1(s1), lc2(s1), rc2(s1)
	int lc1s1 = -1;
	int lc2s1 = -1;
	int rc1s1 = -1;
	int rc2s1 = -1;
	unsigned lc1s1_lab = none_label;
	unsigned lc2s1_lab = none_label;
	unsigned rc1s1_lab = none_label;
	unsigned rc2s1_lab = none_label;
	if(s1 != -1) {
		lc1s1 = stateitem.dep_l1[s1];
		lc2s1 = stateitem.dep_l2[s1];
		rc1s1 = stateitem.dep_r1[s1];
		rc2s1 = stateitem.dep_r2[s1];
		lc1s1_lab = stateitem.arc_l1[s1];
		lc2s1_lab = stateitem.arc_l2[s1];
		rc1s1_lab = stateitem.arc_r1[s1];
		rc2s1_lab = stateitem.arc_r2[s1];
	}
if(DEBUG){
	std::cerr<<"lc1s1:"<<lc1s1<<" lc2s1:"<<lc2s1<<" rc1s1:"<<rc1s1<<" rc2s1:"<<rc2s1;
	std::cerr<<"lc1s1_lab:"<<(lc1s1_lab == none_label? "NONE":ld.convert(lc1s1_lab))<<" " 
		<<"lc2s1_lab:"<<(lc2s1_lab == none_label? "NONE":ld.convert(lc2s1_lab))<<" "
		<<"rc1s1_lab:"<<(rc1s1_lab == none_label? "NONE":ld.convert(rc1s1_lab))<<" "
		<<"rc2s1_lab:"<<(rc2s1_lab == none_label? "NONE":ld.convert(rc2s1_lab));
	std::cerr<<std::endl;
}
	if(lc1s1 != -1) word_features.push_back(i_words[lc1s1]); else word_features.push_back(i_w_none);
        if(lc2s1 != -1) word_features.push_back(i_words[lc2s1]); else word_features.push_back(i_w_none);
        if(rc1s1 != -1) word_features.push_back(i_words[rc1s1]); else word_features.push_back(i_w_none);
        if(rc2s1 != -1) word_features.push_back(i_words[rc2s1]); else word_features.push_back(i_w_none);
	if(lc1s1 != -1) postag_features.push_back(i_postags[lc1s1]); else postag_features.push_back(i_t_none);
        if(lc2s1 != -1) postag_features.push_back(i_postags[lc2s1]); else postag_features.push_back(i_t_none);
        if(rc1s1 != -1) postag_features.push_back(i_postags[rc1s1]); else postag_features.push_back(i_t_none);
        if(rc2s1 != -1) postag_features.push_back(i_postags[rc2s1]); else postag_features.push_back(i_t_none);
        if(lc1s1_lab != none_label) label_features.push_back(i_labels[lc1s1_lab]); else label_features.push_back(i_l_none);
        if(lc2s1_lab != none_label) label_features.push_back(i_labels[lc2s1_lab]); else label_features.push_back(i_l_none);
        if(rc1s1_lab != none_label) label_features.push_back(i_labels[rc1s1_lab]); else label_features.push_back(i_l_none);
        if(rc2s1_lab != none_label) label_features.push_back(i_labels[rc2s1_lab]); else label_features.push_back(i_l_none);
	//lc1(s2), rc1(s2), lc2(s2), rc2(s2)
        int lc1s2 = -1;
        int lc2s2 = -1;
        int rc1s2 = -1;
        int rc2s2 = -1;
	unsigned lc1s2_lab = none_label;
        unsigned lc2s2_lab = none_label;
        unsigned rc1s2_lab = none_label;
        unsigned rc2s2_lab = none_label;
        if(s2 != -1) {
                lc1s2 = stateitem.dep_l1[s2];
                lc2s2 = stateitem.dep_l2[s2];
                rc1s2 = stateitem.dep_r1[s2];
                rc2s2 = stateitem.dep_r2[s2];
		lc1s2_lab = stateitem.arc_l1[s2];
                lc2s2_lab = stateitem.arc_l2[s2];
                rc1s2_lab = stateitem.arc_r1[s2];
                rc2s2_lab = stateitem.arc_r2[s2];
        }
if(DEBUG){
	std::cerr<<"lc1s2:"<<lc1s2<<" lc2s2:"<<lc2s2<<" rc1s2:"<<rc1s2<<" rc2s2:"<<rc2s2;
	std::cerr<<"lc1s2_lab:"<<(lc1s2_lab == none_label? "NONE":ld.convert(lc1s2_lab))<<" "
		<<"lc2s2_lab:"<<(lc2s2_lab == none_label? "NONE":ld.convert(lc2s2_lab))<<" "
		<<"rc1s2_lab:"<<(rc1s2_lab == none_label? "NONE":ld.convert(rc1s2_lab))<<" "
		<<"rc2s2_lab:"<<(rc2s2_lab == none_label? "NONE":ld.convert(rc2s2_lab));
	std::cerr<<std::endl;
}
	if(lc1s2 != -1) word_features.push_back(i_words[lc1s2]); else word_features.push_back(i_w_none);
        if(lc2s2 != -1) word_features.push_back(i_words[lc2s2]); else word_features.push_back(i_w_none);
        if(rc1s2 != -1) word_features.push_back(i_words[rc1s2]); else word_features.push_back(i_w_none);
        if(rc2s2 != -1) word_features.push_back(i_words[rc2s2]); else word_features.push_back(i_w_none);
	if(lc1s2 != -1) postag_features.push_back(i_postags[lc1s2]); else postag_features.push_back(i_t_none);
        if(lc2s2 != -1) postag_features.push_back(i_postags[lc2s2]); else postag_features.push_back(i_t_none);
        if(rc1s2 != -1) postag_features.push_back(i_postags[rc1s2]); else postag_features.push_back(i_t_none);
        if(rc2s2 != -1) postag_features.push_back(i_postags[rc2s2]); else postag_features.push_back(i_t_none);
	if(lc1s2_lab != none_label) label_features.push_back(i_labels[lc1s2_lab]); else label_features.push_back(i_l_none);
        if(lc2s2_lab != none_label) label_features.push_back(i_labels[lc2s2_lab]); else label_features.push_back(i_l_none);
        if(rc1s2_lab != none_label) label_features.push_back(i_labels[rc1s2_lab]); else label_features.push_back(i_l_none);
        if(rc2s2_lab != none_label) label_features.push_back(i_labels[rc2s2_lab]); else label_features.push_back(i_l_none);
	//lc1(lc1(s1)), rc1(rc1(s1))
	int lc1lc1s1 = -1;
	int rc1rc1s1 = -1;
	unsigned lc1lc1s1_lab = none_label;
	unsigned rc1rc1s1_lab = none_label;
	if(lc1s1 != -1) {lc1lc1s1 = stateitem.dep_l1[lc1s1]; lc1lc1s1_lab = stateitem.arc_l1[lc1s1];}
	if(rc1s1 != -1) {rc1rc1s1 = stateitem.dep_r1[rc1s1]; rc1rc1s1_lab = stateitem.arc_r1[rc1s1];}
if(DEBUG){
	std::cerr<<"lc1lc1s1:"<<lc1lc1s1<<" lc1lc1s1_lab:"<<(lc1lc1s1_lab == none_label? "NONE":ld.convert(lc1lc1s1_lab))
		<<" rc1rc1s1:"<<rc1rc1s1<<" rc1rc1s1_lab:"<<(rc1rc1s1_lab == none_label? "NONE":ld.convert(rc1rc1s1_lab))<<std::endl;
}
	if(lc1lc1s1 != -1) word_features.push_back(i_words[lc1lc1s1]); else word_features.push_back(i_w_none);
        if(rc1rc1s1 != -1) word_features.push_back(i_words[rc1rc1s1]); else word_features.push_back(i_w_none);
	if(lc1lc1s1 != -1) postag_features.push_back(i_postags[lc1lc1s1]); else postag_features.push_back(i_t_none);
        if(rc1rc1s1 != -1) postag_features.push_back(i_postags[rc1rc1s1]); else postag_features.push_back(i_t_none);
        if(lc1lc1s1_lab != none_label) label_features.push_back(i_labels[lc1lc1s1_lab]); else label_features.push_back(i_l_none);
        if(rc1rc1s1_lab != none_label) label_features.push_back(i_labels[rc1rc1s1_lab]); else label_features.push_back(i_l_none);
	//lc1(lc1(s2)), rc1(rc1(s2))
        int lc1lc1s2 = -1;
        int rc1rc1s2 = -1;
	unsigned lc1lc1s2_lab = none_label;
	unsigned rc1rc1s2_lab = none_label;
        if(lc1s2 != -1) {lc1lc1s2 = stateitem.dep_l1[lc1s2]; lc1lc1s2_lab = stateitem.arc_l1[lc1s2];}
        if(rc1s2 != -1) {rc1rc1s2 = stateitem.dep_r1[rc1s2]; rc1rc1s2_lab = stateitem.arc_r1[rc1s2];}
if(DEBUG){
	std::cerr<<"lc1lc1s2:"<<lc1lc1s2<<" lc1lc1s2_lab:"<<(lc1lc1s2_lab == none_label? "NONE":ld.convert(lc1lc1s2_lab))
		<<" rc1rc1s2:"<<rc1rc1s2<<" rc1rc1s2_lab:"<<(rc1rc1s2_lab == none_label? "NONE":ld.convert(rc1rc1s2_lab))<<std::endl;
}
	if(lc1lc1s2 != -1) word_features.push_back(i_words[lc1lc1s2]); else word_features.push_back(i_w_none);
        if(rc1rc1s2 != -1) word_features.push_back(i_words[rc1rc1s2]); else word_features.push_back(i_w_none);
	if(lc1lc1s2 != -1) postag_features.push_back(i_postags[lc1lc1s2]); else postag_features.push_back(i_t_none);
        if(rc1rc1s2 != -1) postag_features.push_back(i_postags[rc1rc1s2]); else postag_features.push_back(i_t_none);
        if(lc1lc1s2_lab != none_label) label_features.push_back(i_labels[lc1lc1s2_lab]); else label_features.push_back(i_l_none);
        if(rc1rc1s2_lab != none_label) label_features.push_back(i_labels[rc1rc1s2_lab]); else label_features.push_back(i_l_none);
if(DEBUG){
	std::cerr<<"feature ok"<<std::endl;
	std::cerr<<" word_features size: "<< word_features.size()<<std::endl;
	std::cerr<<" postag_features size: "<< postag_features.size()<<std::endl;
	std::cerr<<" label_features size: "<< label_features.size()<<std::endl;
}
	Expression Sw = concatenate(word_features);
	Expression St = concatenate(postag_features);
	Expression Sl = concatenate(label_features);

	vector<Expression> args = {i_bias, i_w2h, Sw, i_t2h, St, i_l2h, Sl};
if(DEBUG)	std::cerr<<"concatenate ok" <<std::endl;
	Expression h = cube(affine_transform(args));

if(DEBUG)	std::cerr<<"cube hidden ok"<<std::endl;

	vector<unsigned> current_valid_actions;	
	for(unsigned j = 0; j < ad.size(); ++j){
		if(j == 0){ //SHIFT
			if(stateitem.buffer.size() == 0) continue;
		}
		else{ //REDUCE
			if(stateitem.stack.size() < 2) continue;
		}
		current_valid_actions.push_back(j);
	}
	assert(current_valid_actions.size() != 0);
if(DEBUG)	std::cerr<<"current_valid_actions.size() "<<current_valid_actions.size()<<"\n";
if(DEBUG) 	std::cerr<<"best prev " << best_prev <<std::endl;

	vector<Expression> scoree(current_valid_actions.size());
	vector<float> score(current_valid_actions.size());
        for(unsigned j = 0; j < current_valid_actions.size(); ++j){
		Expression lstm_input = concatenate({h, i_actions[current_valid_actions[j]]});
		Expression branch = seq.add_input(RNNPointer(best_prev), lstm_input);

		scoree[j] = affine_transform({i_Sbias, i_W, branch});
		score[j] = as_scalar(cg.incremental_forward(scoree[j]));
		seq.rewind_one_step();
	}
	
	unsigned bestj = 0;
	unsigned besta = current_valid_actions[0];
	float bestscore = score[0];
	for(unsigned j = 1; j < score.size(); j ++){
		if(score[j] >= bestscore) {
			bestscore = score[j]; bestj = j; besta = current_valid_actions[j];
		}
	}

	unsigned goldj = 666;
	unsigned golda = actions[i];
	if(train){
	for(unsigned j = 0; j < current_valid_actions.size(); ++j){
		if(current_valid_actions[j] == actions[i]) {goldj = j; break;}
	}
	assert(goldj != 666);
	}
if(DEBUG)	std::cerr<<"bestj : "<<bestj<<" "<<ad.convert(besta)<<"\n"<<"goldj : "<<goldj<<" "<<ad.convert(golda)<<"\n";
	
	unsigned action = besta;
	if(train){
		if(bestj == goldj) (*cor) ++;
		(*overall) ++;
		action = golda;
	}
	
	if(train) {
		if(goldj != bestj){
			errs.push_back(pairwise_rank_loss(scoree[goldj], scoree[bestj], margin));	
if(DEBUG)		std::cerr<<"loss\n";
		}
		else{
if(DEBUG)		std::cerr<<"no loss\n";
		}
	}
	if(train) best_prev = seq_len + goldj;
	else best_prev = seq_len + bestj;
	seq_len += current_valid_actions.size();
		
	string line = ad.convert(action);
	istrstream istr(line.c_str());
	string action_name, action_label;
	istr>>action_name;
	istr>>action_label;
	
	stateitem.transit(action_name, action);
    }
    if(!train){
    for(unsigned i = 0; i < slen - 1; i ++){
	string word = wd.convert(words[i]);
    	if(valid(word) == false) continue;
	if(stateitem.dep[i] == deps[i]){
		(*cor) ++;
                if(deps[i] == -1 || stateitem.label[i] == labels[i]){
                        (*predict) ++;
                }
	}
	(*overall)++;
    }
    }
    if(errs.size() != 0) return sum(errs);
    else return zeroes(cg, {1});
}
};


int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  WORD_DIM = conf["word_dim"].as<unsigned>();
  POSTAG_DIM = conf["pos_dim"].as<unsigned>();
  LABEL_DIM = conf["rel_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  LSTM_HIDDEN_DIM = conf["lstm_hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["act_dim"].as<unsigned>();
  LAYERS = conf["layers"].as<unsigned>();
  
  unk_prob = conf["unk_prob"].as<float>();
  pdrop = conf["pdrop"].as<float>();
  margin = conf["margin"].as<float>();
 
  DEBUG = conf.count("debug");

  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  assert(pdrop >= 0.); assert(pdrop <= 1.);

  vector<Instance> training,dev,test;
  string line;

  vector<string> poses = {"#",  "$",  "''",  ",",  "-LRB-",  "-RRB-",  ".",  ":",  "CC",  "CD",  "DT",  "EX",  "FW",  "IN",  "JJ",  "JJR",  "JJS",  "LS",  "MD",  "NN",  "NNP",  "NNPS",  "NNS",  "PDT",  "POS",  "PRP",  "PRP$",  "RB",  "RBR",  "RBS",  "RP",  "SYM",  "TO",  "UH",  "VB",  "VBD",  "VBG",  "VBN",  "VBP",  "VBZ",  "WDT",  "WP",  "WP$",  "WRB",  "``"};
  for(unsigned i = 0; i < poses.size(); i ++) td.convert(poses[i]);
  tROOT = td.convert("ROOT");
  td.freeze();

//  vector<string> labels = {"acomp",  "advcl",  "advmod",  "amod",  "appos",  "aux",  "auxpass",  "cc",  "ccomp",  "conj",  "cop",  "csubj",  "csubjpass",  "dep",  "det",  "discourse",  "dobj",  "expl",  "infmod",  "iobj",  "mark",  "mwe",  "neg",  "nn",  "npadvmod",  "nsubj",  "nsubjpass",  "num",  "number",  "parataxis",  "partmod",  "pcomp",  "pobj",  "poss",  "possessive",  "preconj",  "predet",  "prep",  "prt",  "punct",  "quantmod",  "rcmod",  "root",  "tmod",  "xcomp"};
//  none_label= ld.convert("-NONE-");
//  for(int i = 0; i < labels.size(); i ++) ld.convert(labels[i]);
  
  vector<string> leftlabels = {"acomp",  "advcl",  "advmod", "amod", "appos",  "aux",  "auxpass",  "cc",  "ccomp",  "conj",  "cop",  "csubj",  "csubjpass",  "dep",  "det",  "discourse",  "dobj",  "expl", "mark",  "mwe",  "neg",  "nn",  "npadvmod",  "nsubj",  "nsubjpass",  "num",  "number",  "parataxis",  "partmod", "pobj",  "poss",  "possessive",  "preconj",  "predet",  "prep",  "prt",  "punct", "quantmod", "tmod",  "xcomp"};
  vector<string> rightlabels = {"acomp",  "advcl",  "advmod",  "amod",  "appos",  "aux",  "auxpass",  "cc",  "ccomp",  "conj",  "cop", "dep",  "det",  "discourse",  "dobj",  "expl",  "infmod",  "iobj",  "mwe",  "neg",  "nn",  "npadvmod",  "nsubj",  "nsubjpass",  "num",  "number",  "parataxis",  "partmod",  "pcomp",  "pobj",  "poss",  "possessive", "preconj", "prep",  "prt",  "punct",  "quantmod",  "rcmod", "tmod",  "xcomp"};
  
  ad.convert("SHIFT -NONE-");
  none_label = ld.convert("SHIFT -NONE-");
  for(unsigned i = 0; i < leftlabels.size(); i ++) {ad.convert("REDUCE_LEFT "+leftlabels[i]); ld.convert("REDUCE_LEFT "+leftlabels[i]);}
  for(unsigned i = 0; i < rightlabels.size(); i ++) {ad.convert("REDUCE_RIGHT "+rightlabels[i]); ld.convert("REDUCE_RIGHT "+rightlabels[i]);}
  kUNK = wd.convert("*UNK*");
  lROOT = ld.convert("ROOT");
  ad.freeze();
  ld.freeze();

  //reading pretrained
  //reading training data
  cerr << "Loading from " << conf["training_data"].as<string>() << "as training data : ";
  {
    ifstream in(conf["training_data"].as<string>().c_str());
    assert(in);
    while(getline(in, line)) {
      Instance instance;
      instance.load(line);
      training.push_back(instance);	
    }
    cerr<<training.size()<<"\n";
  }

  //couting
  set<unsigned> training_vocab;
  set<unsigned> singletons;
  {
    map<unsigned, unsigned> counts;
    for (auto& sent : training){
      vector<unsigned>& words = sent.words;
      for (unsigned i = 0; i < sent.size(); ++i){
        training_vocab.insert(words[i]); counts[words[i]]++;
      }
    }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);

    cerr<<"the training word dict size is " << training_vocab.size()
           << " where The singletons have " << singletons.size() << "\n";
  }

  //replace unk 
  {
    int unk = 0;
    int total = 0;
    for(auto& sent : training){
      for(auto& w : sent.words){
        if(singletons.count(w) && dynet::rand01() < unk_prob) {w = kUNK; unk += 1;}
        total += 1;
      }
    }
    if(total ==0) total = 1;
    cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
  }

  //reading dev data 
  if(conf.count("dev_data")){
    cerr << "Loading from " << conf["dev_data"].as<string>() << "as dev data : ";
    ifstream in(conf["dev_data"].as<string>().c_str());
    string line;
    while(getline(in,line)){
        Instance inst;
        inst.load(line);
        dev.push_back(inst);
    }
    cerr<<dev.size()<<"\n";
  }

  //replace unk
  {
    int unk = 0;
    int total = 0;
    for(auto& sent : dev){
      vector<unsigned>& words = sent.words;
      for(unsigned i = 0; i < sent.size(); i ++){
        if(training_vocab.count(words[i]) == 0){
              words[i] = kUNK;
              unk += 1;
        }
        total += 1;
      }
    }
    cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
  }
  //reading test data
  if(conf.count("test_data")){
    cerr << "Loading from " << conf["test_data"].as<string>() << "as test data : ";
    ifstream in(conf["test_data"].as<string>().c_str());
    string line;
    while(getline(in,line)){
        Instance inst;
        inst.load(line);
        test.push_back(inst);
    }
    cerr<<test.size()<<"\n";
  }

  //replace unk
  {
    int unk = 0;
    int total = 0;
    for(auto& sent : test){
      vector<unsigned>& words = sent.words;
      for(unsigned i = 0; i < sent.size(); i ++){
        if(training_vocab.count(words[i]) == 0){
              words[i] = kUNK;
              unk += 1;
        }
        total += 1;
      }
    }
    cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
  }
 
  VOCAB_SIZE = wd.size();
  LABEL_SIZE = ld.size();
  POSTAG_SIZE = td.size();
  ACTION_SIZE = ad.size();

  ///////////////////////////******************************
  ostringstream os;
  os << "ordered"
     << '_' << WORD_DIM
     << '_' << POSTAG_DIM
     << '_' << LABEL_DIM
     << '_' << HIDDEN_DIM
     << '_' << unk_prob
     << '_' << pdrop
     << "-pid" << getpid() << ".params";

  const string fname = os.str();
  cerr << "Parameter will be written to: " << fname << endl;
  double best = -9e+99;

  Model model;
  Trainer* sgd = nullptr;
  unsigned method = conf["train_methods"].as<unsigned>();
  if(method == 0)
        sgd = new SimpleSGDTrainer(&model,0.1, 0.1);
  else if(method == 1)
        sgd = new MomentumSGDTrainer(&model,0.01, 0.9, 0.1);
  else if(method == 2){
        sgd = new AdagradTrainer(&model);
        sgd->clipping_enabled = false;
  }
  else if(method == 3){
        sgd = new AdamTrainer(&model);
        sgd->clipping_enabled = false;
  }
  RNNLanguageModel lm(model);
  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (conf.count("model")) {
    string fname = conf["model"].as<string>();
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  if(conf.count("train")){

  unsigned report_every_i = conf["report_i"].as<unsigned>();
  unsigned dev_every_i_reports = conf["dev_report_i"].as<unsigned>();
  unsigned si = training.size();
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    double tpredict = 0;
    unsigned toverall = 0;
    double correct = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      
      ComputationGraph cg;
      if (si == training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      auto& sent = training[order[si]];
      ++si;
      Expression losse = lm.BuildTaggingGraph(sent, cg, &correct, &tpredict, &toverall, true);
      loss += as_scalar(cg.forward(losse));
      cg.backward(losse);
      sgd->update(1.0);
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / toverall) << " ppl=" << exp(loss / toverall) << " (acc=" << (correct / toverall) << ") ";

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dpredict = 0;
      unsigned doverall = 0;
      double dcorr = 0;
      int i = 0;
      for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildTaggingGraph(sent, cg, &dcorr, &dpredict, &doverall, false);
        i += 1;
      }
      if ((dcorr/doverall) > best) {
        best = (dcorr/doverall);
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] UAS = " << (dcorr/doverall) << " LAS = " << (dpredict/doverall);
    }
  }
  delete sgd;
  }
  else{
      double dpredict = 0;
      unsigned doverall = 0;
      double dcorr = 0;
      int i = 0;
      for (auto& sent : test) {
        ComputationGraph cg;
        lm.BuildTaggingGraph(sent, cg, &dcorr, &dpredict, &doverall, false);
        i += 1;
      }
      cerr << "\n***TEST UAS = " << (dcorr/doverall) << " LAS = " << (dpredict/doverall);
 
  }
}

