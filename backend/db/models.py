from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    labels = relationship("Label", back_populates="owner")

class Material(Base):
    __tablename__ = "materials"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    Eg_0K_eV = Column(Float, nullable=False)
    Alpha_evK = Column(Float, nullable=False)
    Beta_K = Column(Float, nullable=False)
    me_eff = Column(Float, nullable=False)
    mh_eff = Column(Float, nullable=False)
    epsilon_r = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    labels = relationship(
        "Label",
        secondary="material_labels",
        back_populates="materials"
    )

class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    label = Column(String, nullable=False)
    owner = relationship("User", back_populates="labels")

    materials = relationship(
        "Material",
        secondary="material_labels",
        back_populates="labels"
    )

class MaterialLabel(Base):
    __tablename__ = "material_labels"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False)
    label_id = Column(Integer, ForeignKey("labels.id"), nullable=False)