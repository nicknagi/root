/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/Browsable/RItem.hxx>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RMiniFile.hxx>

#include "TClass.h"

using namespace std::string_literals;

using namespace ROOT::Experimental::Browsable;


// ==============================================================================================

class RFieldElement : public RElement {
protected:
   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNTuple;

   ROOT::Experimental::DescriptorId_t fFieldId;

public:

   RFieldElement(std::shared_ptr<ROOT::Experimental::RNTupleReader> tuple, const ROOT::Experimental::DescriptorId_t id) : RElement(), fNTuple(tuple), fFieldId(id) {}

   virtual ~RFieldElement() = default;

   /** Name of NTuple */
   std::string GetName() const override { return fNTuple->GetDescriptor().GetFieldDescriptor(fFieldId).GetFieldName(); }

   /** Title of NTuple */
   std::string GetTitle() const override { return "RField title"s; }

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   const TClass *GetClass() const { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }
};

// ==============================================================================================

/** \class RNTupleElement
\ingroup rbrowser
\brief Browsing of RNTuple
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleElement : public RElement {
protected:
   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNTuple;

public:
   RNTupleElement(const std::string &tuple_name, const std::string &filename);

   virtual ~RNTupleElement() = default;

   /** Returns true if no ntuple found */
   bool IsNull() const { return !fNTuple; }

   /** Name of NTuple */
   std::string GetName() const override { return fNTuple->GetDescriptor().GetName(); }

   /** Title of NTuple */
   std::string GetTitle() const override { return "RNTuple title"s; }

   /** Create iterator for childs elements if any */
   std::unique_ptr<RLevelIter> GetChildsIter() override;

   const TClass *GetClass() const { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }

   //EActionKind GetDefaultAction() const override;

   //bool IsCapable(EActionKind) const override;
};


RNTupleElement::RNTupleElement(const std::string &tuple_name, const std::string &filename)
{
   fNTuple = ROOT::Experimental::RNTupleReader::Open(tuple_name, filename);
}


// ==============================================================================================



class RFieldsIterator : public RLevelIter {

   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNTuple;
   std::vector<ROOT::Experimental::DescriptorId_t> fFieldIds;
   int fCounter{-1};

public:
   RFieldsIterator(std::shared_ptr<ROOT::Experimental::RNTupleReader> tuple, std::vector<ROOT::Experimental::DescriptorId_t> &&ids) : fNTuple(tuple), fFieldIds(ids)
   {
   }

   virtual ~RFieldsIterator() = default;

   bool Next() override
   {
      return ++fCounter < (int) fFieldIds.size();
   }

   std::string GetItemName() const override
   {
      return fNTuple->GetDescriptor().GetFieldDescriptor(fFieldIds[fCounter]).GetFieldName();
   }

   bool CanItemHaveChilds() const override
   {
      auto subrange = fNTuple->GetDescriptor().GetFieldRange(fFieldIds[fCounter]);
      return subrange.begin() != subrange.end();
   }

   /** Create element for the browser */
   std::unique_ptr<RItem> CreateItem() override
   {

      int nchilds = 0;
      for (auto &sub: fNTuple->GetDescriptor().GetFieldRange(fFieldIds[fCounter])) { (void) sub; nchilds++; }

      auto &field = fNTuple->GetDescriptor().GetFieldDescriptor(fFieldIds[fCounter]);

      auto item = std::make_unique<RItem>(field.GetFieldName(), nchilds, "sap-icon://measuring-point");

      item->SetTitle("RField title");

      return item;
   }

   std::shared_ptr<RElement> GetElement() override
   {
      return std::make_shared<RFieldElement>(fNTuple, fFieldIds[fCounter]);
   }

};


std::unique_ptr<RLevelIter> RFieldElement::GetChildsIter()
{
   std::vector<ROOT::Experimental::DescriptorId_t> ids;

   for (auto &f : fNTuple->GetDescriptor().GetFieldRange(fFieldId))
      ids.emplace_back(f.GetId());

   if (ids.size() == 0) return nullptr;
   return std::make_unique<RFieldsIterator>(fNTuple, std::move(ids));
}

std::unique_ptr<RLevelIter> RNTupleElement::GetChildsIter()
{
   std::vector<ROOT::Experimental::DescriptorId_t> ids;

   for (auto &f : fNTuple->GetDescriptor().GetTopLevelFields())
      ids.emplace_back(f.GetId());

   if (ids.size() == 0) return nullptr;
   return std::make_unique<RFieldsIterator>(fNTuple, std::move(ids));
}


// ==============================================================================================

/** \class RNTupleProvider
\ingroup rbrowser
\brief Provider for RNTuple classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RNTupleProvider : public RProvider {

public:

   RNTupleProvider()
   {
      RegisterNTupleFunc([](const std::string &tuple_name, const std::string &filename) -> std::shared_ptr<RElement> {
         auto elem = std::make_shared<RNTupleElement>(tuple_name, filename);
         return elem->IsNull() ? nullptr : elem;
      });
   }

   virtual ~RNTupleProvider()
   {
      RegisterNTupleFunc(nullptr);
   }

} newRNTupleProvider;

