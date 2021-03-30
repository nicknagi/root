#ifndef TMVA_SOFIE_ROPERATOR_ADD
#define TMVA_SOFIE_ROPERATOR_ADD

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Add final : public ROperator
{

private:

   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Add() = delete;
   ROperator_Add(std::string nameA, std::string nameB, std::string nameY):
      fNX1(UTILITY::Clean_name(nameA)), fNX2(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      if (input.size() != 2) throw std::runtime_error("TMVA SOFIE Add op only needs 2 input tensors");
      for (auto& i: input){
         if (i.size() > 2){
            throw std::runtime_error("TMVA SOFIE Add Op Only supports 2D tensors");
         }
      }
      auto ret = input; //assuming both are equal
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX1) == false || model.CheckIfTensorAlreadyExist(fNX2) == false) {   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Add Op Input Tensor is not found in model");
      }
      std::vector<size_t> fShapeX1 = model.GetTensorShape(fNX1);
      std::vector<size_t> fShapeX2 = model.GetTensorShape(fNX2);

      // Assumption the tensor to broadcast has already been initialized
      if (fShapeX1.size() < fShapeX2.size()) {
         auto original_data = model.GetInitializedTensorData(fNX1);
         std::shared_ptr<void> new_data_ptr(UTILITY::Unidirectional_broadcast<float>(static_cast<float*>(original_data.get()), fShapeX1, fShapeX2), std::default_delete<float[]>());
         model.UpdateInitializedTensor(fNX1, model.GetTensorType(fNX1), fShapeX2, new_data_ptr);
         fShapeX1 = fShapeX2;
      } else if(fShapeX2.size() < fShapeX1.size()) {
         auto original_data = model.GetInitializedTensorData(fNX2);
         std::shared_ptr<void> new_data_ptr(UTILITY::Unidirectional_broadcast<float>(static_cast<float*>(original_data.get()), fShapeX2, fShapeX1), std::default_delete<float[]>());
         model.UpdateInitializedTensor(fNX2, model.GetTensorType(fNX2), fShapeX1, new_data_ptr);
         fShapeX2 = fShapeX1;
      }
      fShape = fShapeX1;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX1), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Transpose Add called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t" << "tensor_" << fNY << "[id] = tensor_" << fNX1 << "[id] + tensor_" << fNX2 << "[id];\n";
      out << "\t}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif
