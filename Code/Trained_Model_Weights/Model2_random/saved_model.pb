û<
ß
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.22v2.8.2-0-g2ea19cbb5758;
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
º
/bidirectional_1/forward_gru_1/gru_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/bidirectional_1/forward_gru_1/gru_cell_4/kernel
³
Cbidirectional_1/forward_gru_1/gru_cell_4/kernel/Read/ReadVariableOpReadVariableOp/bidirectional_1/forward_gru_1/gru_cell_4/kernel*
_output_shapes

:*
dtype0
Î
9bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*J
shared_name;9bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel
Ç
Mbidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp9bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel*
_output_shapes

:
*
dtype0
¶
-bidirectional_1/forward_gru_1/gru_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-bidirectional_1/forward_gru_1/gru_cell_4/bias
¯
Abidirectional_1/forward_gru_1/gru_cell_4/bias/Read/ReadVariableOpReadVariableOp-bidirectional_1/forward_gru_1/gru_cell_4/bias*
_output_shapes

:*
dtype0
¼
0bidirectional_1/backward_gru_1/gru_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20bidirectional_1/backward_gru_1/gru_cell_5/kernel
µ
Dbidirectional_1/backward_gru_1/gru_cell_5/kernel/Read/ReadVariableOpReadVariableOp0bidirectional_1/backward_gru_1/gru_cell_5/kernel*
_output_shapes

:*
dtype0
Ð
:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*K
shared_name<:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel
É
Nbidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel*
_output_shapes

:
*
dtype0
¸
.bidirectional_1/backward_gru_1/gru_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*?
shared_name0.bidirectional_1/backward_gru_1/gru_cell_5/bias
±
Bbidirectional_1/backward_gru_1/gru_cell_5/bias/Read/ReadVariableOpReadVariableOp.bidirectional_1/backward_gru_1/gru_cell_5/bias*
_output_shapes

:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
È
6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/m
Á
JAdam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/m*
_output_shapes

:*
dtype0
Ü
@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*Q
shared_nameB@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/m
Õ
TAdam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/m*
_output_shapes

:
*
dtype0
Ä
4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/m
½
HAdam/bidirectional_1/forward_gru_1/gru_cell_4/bias/m/Read/ReadVariableOpReadVariableOp4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/m*
_output_shapes

:*
dtype0
Ê
7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*H
shared_name97Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/m
Ã
KAdam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/m*
_output_shapes

:*
dtype0
Þ
AAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*R
shared_nameCAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/m
×
UAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/m*
_output_shapes

:
*
dtype0
Æ
5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/m
¿
IAdam/bidirectional_1/backward_gru_1/gru_cell_5/bias/m/Read/ReadVariableOpReadVariableOp5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/m*
_output_shapes

:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
È
6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/v
Á
JAdam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/v*
_output_shapes

:*
dtype0
Ü
@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*Q
shared_nameB@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/v
Õ
TAdam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/v*
_output_shapes

:
*
dtype0
Ä
4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/v
½
HAdam/bidirectional_1/forward_gru_1/gru_cell_4/bias/v/Read/ReadVariableOpReadVariableOp4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/v*
_output_shapes

:*
dtype0
Ê
7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*H
shared_name97Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/v
Ã
KAdam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/v*
_output_shapes

:*
dtype0
Þ
AAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*R
shared_nameCAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/v
×
UAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/v*
_output_shapes

:
*
dtype0
Æ
5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/v
¿
IAdam/bidirectional_1/backward_gru_1/gru_cell_5/bias/v/Read/ReadVariableOpReadVariableOp5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/v*
_output_shapes

:*
dtype0

NoOpNoOp
ôO
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¯O
value¥OB¢O BO
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
·
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*

%iter

&beta_1

'beta_2
	(decay
)learning_ratemmmm*m+m,m-m.m/mvvvv*v+v,v-v.v/v*
J
*0
+1
,2
-3
.4
/5
6
7
8
9*
J
*0
+1
,2
-3
.4
/5
6
7
8
9*
* 
°
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

5serving_default* 
Á
6cell
7
state_spec
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<_random_generator
=__call__
*>&call_and_return_all_conditional_losses*
Á
?cell
@
state_spec
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E_random_generator
F__call__
*G&call_and_return_all_conditional_losses*
.
*0
+1
,2
-3
.4
/5*
.
*0
+1
,2
-3
.4
/5*
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/bidirectional_1/forward_gru_1/gru_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE9bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-bidirectional_1/forward_gru_1/gru_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0bidirectional_1/backward_gru_1/gru_cell_5/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.bidirectional_1/backward_gru_1/gru_cell_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

W0
X1*
* 
* 
* 
Ó

*kernel
+recurrent_kernel
,bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]_random_generator
^__call__
*_&call_and_return_all_conditional_losses*
* 

*0
+1
,2*

*0
+1
,2*
	
`0* 


astates
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
8	variables
9trainable_variables
:regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
Ó

-kernel
.recurrent_kernel
/bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k_random_generator
l__call__
*m&call_and_return_all_conditional_losses*
* 

-0
.1
/2*

-0
.1
/2*
	
n0* 


ostates
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	utotal
	vcount
w	variables
x	keras_api*
H
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api*

*0
+1
,2*

*0
+1
,2*
	
`0* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

60*
* 
* 
* 

-0
.1
/2*

-0
.1
/2*
	
n0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
g	variables
htrainable_variables
iregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

?0*
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

w	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

y0
z1*

|	variables*
* 
* 
* 
	
`0* 
* 
* 
* 
* 
	
n0* 
* 
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

%serving_default_bidirectional_1_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
È
StatefulPartitionedCallStatefulPartitionedCall%serving_default_bidirectional_1_input-bidirectional_1/forward_gru_1/gru_cell_4/bias/bidirectional_1/forward_gru_1/gru_cell_4/kernel9bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel.bidirectional_1/backward_gru_1/gru_cell_5/bias0bidirectional_1/backward_gru_1/gru_cell_5/kernel:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kerneldense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_43947
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpCbidirectional_1/forward_gru_1/gru_cell_4/kernel/Read/ReadVariableOpMbidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/Read/ReadVariableOpAbidirectional_1/forward_gru_1/gru_cell_4/bias/Read/ReadVariableOpDbidirectional_1/backward_gru_1/gru_cell_5/kernel/Read/ReadVariableOpNbidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/Read/ReadVariableOpBbidirectional_1/backward_gru_1/gru_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOpJAdam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/m/Read/ReadVariableOpTAdam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/m/Read/ReadVariableOpHAdam/bidirectional_1/forward_gru_1/gru_cell_4/bias/m/Read/ReadVariableOpKAdam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/m/Read/ReadVariableOpUAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/m/Read/ReadVariableOpIAdam/bidirectional_1/backward_gru_1/gru_cell_5/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpJAdam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/v/Read/ReadVariableOpTAdam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/v/Read/ReadVariableOpHAdam/bidirectional_1/forward_gru_1/gru_cell_4/bias/v/Read/ReadVariableOpKAdam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/v/Read/ReadVariableOpUAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/v/Read/ReadVariableOpIAdam/bidirectional_1/backward_gru_1/gru_cell_5/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_47129

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate/bidirectional_1/forward_gru_1/gru_cell_4/kernel9bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel-bidirectional_1/forward_gru_1/gru_cell_4/bias0bidirectional_1/backward_gru_1/gru_cell_5/kernel:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel.bidirectional_1/backward_gru_1/gru_cell_5/biastotalcounttotal_1count_1Adam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/m@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/m4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/m7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/mAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/m5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v6Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/v@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/v4Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/v7Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/vAAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/v5Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_47256«½9
ª,
ï
G__inference_sequential_1_layer_call_and_return_conditional_losses_43062

inputs'
bidirectional_1_43026:'
bidirectional_1_43028:'
bidirectional_1_43030:
'
bidirectional_1_43032:'
bidirectional_1_43034:'
bidirectional_1_43036:

dense_2_43039:
dense_2_43041:
dense_3_43044:
dense_3_43046:
identity¢'bidirectional_1/StatefulPartitionedCall¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallí
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_1_43026bidirectional_1_43028bidirectional_1_43030bidirectional_1_43032bidirectional_1_43034bidirectional_1_43036*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42981
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_43039dense_2_43041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42565
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_43044dense_3_43046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42582§
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_43028*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_43034*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOp(^bidirectional_1/StatefulPartitionedCallS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

/__inference_bidirectional_1_layer_call_fn_44027

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42981o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ(
©
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41144

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
÷(
ª
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_46843

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
û
ð
__inference_loss_fn_0_46854l
Zbidirectional_1_forward_gru_1_gru_cell_4_kernel_regularizer_square_readvariableop_resource:
identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpì
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpZbidirectional_1_forward_gru_1_gru_cell_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
IdentityIdentityCbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp
ëÖ

J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44981

inputsB
0forward_gru_1_gru_cell_4_readvariableop_resource:I
7forward_gru_1_gru_cell_4_matmul_readvariableop_resource:K
9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
C
1backward_gru_1_gru_cell_5_readvariableop_resource:J
8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:L
:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:

identity¢/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢(backward_gru_1/gru_cell_5/ReadVariableOp¢backward_gru_1/while¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢'forward_gru_1/gru_cell_4/ReadVariableOp¢forward_gru_1/whileI
forward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:k
!forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru_1/strided_sliceStridedSliceforward_gru_1/Shape:output:0*forward_gru_1/strided_slice/stack:output:0,forward_gru_1/strided_slice/stack_1:output:0,forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru_1/zeros/packedPack$forward_gru_1/strided_slice:output:0%forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru_1/zerosFill#forward_gru_1/zeros/packed:output:0"forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru_1/transpose	Transposeinputs%forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
forward_gru_1/Shape_1Shapeforward_gru_1/transpose:y:0*
T0*
_output_shapes
:m
#forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
forward_gru_1/strided_slice_1StridedSliceforward_gru_1/Shape_1:output:0,forward_gru_1/strided_slice_1/stack:output:0.forward_gru_1/strided_slice_1/stack_1:output:0.forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
forward_gru_1/TensorArrayV2TensorListReserve2forward_gru_1/TensorArrayV2/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
5forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru_1/transpose:y:0Lforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
forward_gru_1/strided_slice_2StridedSliceforward_gru_1/transpose:y:0,forward_gru_1/strided_slice_2/stack:output:0.forward_gru_1/strided_slice_2/stack_1:output:0.forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp0forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0
 forward_gru_1/gru_cell_4/unstackUnpack/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0»
forward_gru_1/gru_cell_4/MatMulMatMul&forward_gru_1/strided_slice_2:output:06forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 forward_gru_1/gru_cell_4/BiasAddBiasAdd)forward_gru_1/gru_cell_4/MatMul:product:0)forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
forward_gru_1/gru_cell_4/splitSplit1forward_gru_1/gru_cell_4/split/split_dim:output:0)forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitª
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0µ
!forward_gru_1/gru_cell_4/MatMul_1MatMulforward_gru_1/zeros:output:08forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
"forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd+forward_gru_1/gru_cell_4/MatMul_1:product:0)forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿu
*forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
 forward_gru_1/gru_cell_4/split_1SplitV+forward_gru_1/gru_cell_4/BiasAdd_1:output:0'forward_gru_1/gru_cell_4/Const:output:03forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split«
forward_gru_1/gru_cell_4/addAddV2'forward_gru_1/gru_cell_4/split:output:0)forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru_1/gru_cell_4/SigmoidSigmoid forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
forward_gru_1/gru_cell_4/add_1AddV2'forward_gru_1/gru_cell_4/split:output:1)forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid"forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
forward_gru_1/gru_cell_4/mulMul&forward_gru_1/gru_cell_4/Sigmoid_1:y:0)forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
forward_gru_1/gru_cell_4/add_2AddV2'forward_gru_1/gru_cell_4/split:output:2 forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru_1/gru_cell_4/ReluRelu"forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru_1/gru_cell_4/mul_1Mul$forward_gru_1/gru_cell_4/Sigmoid:y:0forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
forward_gru_1/gru_cell_4/subSub'forward_gru_1/gru_cell_4/sub/x:output:0$forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
forward_gru_1/gru_cell_4/mul_2Mul forward_gru_1/gru_cell_4/sub:z:0+forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
forward_gru_1/gru_cell_4/add_3AddV2"forward_gru_1/gru_cell_4/mul_1:z:0"forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
+forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   â
forward_gru_1/TensorArrayV2_1TensorListReserve4forward_gru_1/TensorArrayV2_1/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
forward_gru_1/whileWhile)forward_gru_1/while/loop_counter:output:0/forward_gru_1/while/maximum_iterations:output:0forward_gru_1/time:output:0&forward_gru_1/TensorArrayV2_1:handle:0forward_gru_1/zeros:output:0&forward_gru_1/strided_slice_1:output:0Eforward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00forward_gru_1_gru_cell_4_readvariableop_resource7forward_gru_1_gru_cell_4_matmul_readvariableop_resource9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
forward_gru_1_while_body_44727**
cond"R 
forward_gru_1_while_cond_44726*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
>forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ì
0forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru_1/while:output:3Gforward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0v
#forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
forward_gru_1/strided_slice_3StridedSlice9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,forward_gru_1/strided_slice_3/stack:output:0.forward_gru_1/strided_slice_3/stack_1:output:0.forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masks
forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          À
forward_gru_1/transpose_1	Transpose9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    J
backward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:l
"backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru_1/strided_sliceStridedSlicebackward_gru_1/Shape:output:0+backward_gru_1/strided_slice/stack:output:0-backward_gru_1/strided_slice/stack_1:output:0-backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
 
backward_gru_1/zeros/packedPack%backward_gru_1/strided_slice:output:0&backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru_1/zerosFill$backward_gru_1/zeros/packed:output:0#backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru_1/transpose	Transposeinputs&backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
backward_gru_1/Shape_1Shapebackward_gru_1/transpose:y:0*
T0*
_output_shapes
:n
$backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
backward_gru_1/strided_slice_1StridedSlicebackward_gru_1/Shape_1:output:0-backward_gru_1/strided_slice_1/stack:output:0/backward_gru_1/strided_slice_1/stack_1:output:0/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
backward_gru_1/TensorArrayV2TensorListReserve3backward_gru_1/TensorArrayV2/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¡
backward_gru_1/ReverseV2	ReverseV2backward_gru_1/transpose:y:0&backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!backward_gru_1/ReverseV2:output:0Mbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
backward_gru_1/strided_slice_2StridedSlicebackward_gru_1/transpose:y:0-backward_gru_1/strided_slice_2/stack:output:0/backward_gru_1/strided_slice_2/stack_1:output:0/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOp1backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0
!backward_gru_1/gru_cell_5/unstackUnpack0backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¨
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¾
 backward_gru_1/gru_cell_5/MatMulMatMul'backward_gru_1/strided_slice_2:output:07backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!backward_gru_1/gru_cell_5/BiasAddBiasAdd*backward_gru_1/gru_cell_5/MatMul:product:0*backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
)backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
backward_gru_1/gru_cell_5/splitSplit2backward_gru_1/gru_cell_5/split/split_dim:output:0*backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¬
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¸
"backward_gru_1/gru_cell_5/MatMul_1MatMulbackward_gru_1/zeros:output:09backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd,backward_gru_1/gru_cell_5/MatMul_1:product:0*backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿv
+backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ®
!backward_gru_1/gru_cell_5/split_1SplitV,backward_gru_1/gru_cell_5/BiasAdd_1:output:0(backward_gru_1/gru_cell_5/Const:output:04backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split®
backward_gru_1/gru_cell_5/addAddV2(backward_gru_1/gru_cell_5/split:output:0*backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru_1/gru_cell_5/SigmoidSigmoid!backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
backward_gru_1/gru_cell_5/add_1AddV2(backward_gru_1/gru_cell_5/split:output:1*backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid#backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
backward_gru_1/gru_cell_5/mulMul'backward_gru_1/gru_cell_5/Sigmoid_1:y:0*backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
backward_gru_1/gru_cell_5/add_2AddV2(backward_gru_1/gru_cell_5/split:output:2!backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru_1/gru_cell_5/ReluRelu#backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru_1/gru_cell_5/mul_1Mul%backward_gru_1/gru_cell_5/Sigmoid:y:0backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
backward_gru_1/gru_cell_5/subSub(backward_gru_1/gru_cell_5/sub/x:output:0%backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
backward_gru_1/gru_cell_5/mul_2Mul!backward_gru_1/gru_cell_5/sub:z:0,backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
backward_gru_1/gru_cell_5/add_3AddV2#backward_gru_1/gru_cell_5/mul_1:z:0#backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
,backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   å
backward_gru_1/TensorArrayV2_1TensorListReserve5backward_gru_1/TensorArrayV2_1/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü
backward_gru_1/whileWhile*backward_gru_1/while/loop_counter:output:00backward_gru_1/while/maximum_iterations:output:0backward_gru_1/time:output:0'backward_gru_1/TensorArrayV2_1:handle:0backward_gru_1/zeros:output:0'backward_gru_1/strided_slice_1:output:0Fbackward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01backward_gru_1_gru_cell_5_readvariableop_resource8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
backward_gru_1_while_body_44878*+
cond#R!
backward_gru_1_while_cond_44877*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
?backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ï
1backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru_1/while:output:3Hbackward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0w
$backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
backward_gru_1/strided_slice_3StridedSlice:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0-backward_gru_1/strided_slice_3/stack:output:0/backward_gru_1/strided_slice_3/stack_1:output:0/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskt
backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
backward_gru_1/transpose_1	Transpose:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0(backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :´
concatConcatV2&forward_gru_1/strided_slice_3:output:0'backward_gru_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ë
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp0^backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2^backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp)^backward_gru_1/gru_cell_5/ReadVariableOp^backward_gru_1/whileS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp/^forward_gru_1/gru_cell_4/MatMul/ReadVariableOp1^forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp(^forward_gru_1/gru_cell_4/ReadVariableOp^forward_gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2b
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2f
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2T
(backward_gru_1/gru_cell_5/ReadVariableOp(backward_gru_1/gru_cell_5/ReadVariableOp2,
backward_gru_1/whilebackward_gru_1/while2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2`
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2d
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2R
'forward_gru_1/gru_cell_4/ReadVariableOp'forward_gru_1/gru_cell_4/ReadVariableOp2*
forward_gru_1/whileforward_gru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ñ	
<sequential_1_bidirectional_1_backward_gru_1_while_cond_40598t
psequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_loop_counterz
vsequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_maximum_iterationsA
=sequential_1_bidirectional_1_backward_gru_1_while_placeholderC
?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_1C
?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_2v
rsequential_1_bidirectional_1_backward_gru_1_while_less_sequential_1_bidirectional_1_backward_gru_1_strided_slice_1
sequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_cond_40598___redundant_placeholder0
sequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_cond_40598___redundant_placeholder1
sequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_cond_40598___redundant_placeholder2
sequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_cond_40598___redundant_placeholder3>
:sequential_1_bidirectional_1_backward_gru_1_while_identity

6sequential_1/bidirectional_1/backward_gru_1/while/LessLess=sequential_1_bidirectional_1_backward_gru_1_while_placeholderrsequential_1_bidirectional_1_backward_gru_1_while_less_sequential_1_bidirectional_1_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: £
:sequential_1/bidirectional_1/backward_gru_1/while/IdentityIdentity:sequential_1/bidirectional_1/backward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "
:sequential_1_bidirectional_1_backward_gru_1_while_identityCsequential_1/bidirectional_1/backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
þ(
©
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41299

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
ë

Â
backward_gru_1_while_cond_42877:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_2<
8backward_gru_1_while_less_backward_gru_1_strided_slice_1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42877___redundant_placeholder0Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42877___redundant_placeholder1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42877___redundant_placeholder2Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42877___redundant_placeholder3!
backward_gru_1_while_identity

backward_gru_1/while/LessLess backward_gru_1_while_placeholder8backward_gru_1_while_less_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: i
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
¾

'__inference_dense_2_layer_call_fn_45308

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¡
while_body_41353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_5_41375_0:*
while_gru_cell_5_41377_0:*
while_gru_cell_5_41379_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_5_41375:(
while_gru_cell_5_41377:(
while_gru_cell_5_41379:
¢(while/gru_cell_5/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_5_41375_0while_gru_cell_5_41377_0while_gru_cell_5_41379_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41299Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_5_41375while_gru_cell_5_41375_0"2
while_gru_cell_5_41377while_gru_cell_5_41377_0"2
while_gru_cell_5_41379while_gru_cell_5_41379_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2T
(while/gru_cell_5/StatefulPartitionedCall(while/gru_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ëÖ

J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42540

inputsB
0forward_gru_1_gru_cell_4_readvariableop_resource:I
7forward_gru_1_gru_cell_4_matmul_readvariableop_resource:K
9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
C
1backward_gru_1_gru_cell_5_readvariableop_resource:J
8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:L
:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:

identity¢/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢(backward_gru_1/gru_cell_5/ReadVariableOp¢backward_gru_1/while¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢'forward_gru_1/gru_cell_4/ReadVariableOp¢forward_gru_1/whileI
forward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:k
!forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru_1/strided_sliceStridedSliceforward_gru_1/Shape:output:0*forward_gru_1/strided_slice/stack:output:0,forward_gru_1/strided_slice/stack_1:output:0,forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru_1/zeros/packedPack$forward_gru_1/strided_slice:output:0%forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru_1/zerosFill#forward_gru_1/zeros/packed:output:0"forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru_1/transpose	Transposeinputs%forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
forward_gru_1/Shape_1Shapeforward_gru_1/transpose:y:0*
T0*
_output_shapes
:m
#forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
forward_gru_1/strided_slice_1StridedSliceforward_gru_1/Shape_1:output:0,forward_gru_1/strided_slice_1/stack:output:0.forward_gru_1/strided_slice_1/stack_1:output:0.forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
forward_gru_1/TensorArrayV2TensorListReserve2forward_gru_1/TensorArrayV2/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
5forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru_1/transpose:y:0Lforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
forward_gru_1/strided_slice_2StridedSliceforward_gru_1/transpose:y:0,forward_gru_1/strided_slice_2/stack:output:0.forward_gru_1/strided_slice_2/stack_1:output:0.forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp0forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0
 forward_gru_1/gru_cell_4/unstackUnpack/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0»
forward_gru_1/gru_cell_4/MatMulMatMul&forward_gru_1/strided_slice_2:output:06forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 forward_gru_1/gru_cell_4/BiasAddBiasAdd)forward_gru_1/gru_cell_4/MatMul:product:0)forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
forward_gru_1/gru_cell_4/splitSplit1forward_gru_1/gru_cell_4/split/split_dim:output:0)forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitª
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0µ
!forward_gru_1/gru_cell_4/MatMul_1MatMulforward_gru_1/zeros:output:08forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
"forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd+forward_gru_1/gru_cell_4/MatMul_1:product:0)forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿu
*forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
 forward_gru_1/gru_cell_4/split_1SplitV+forward_gru_1/gru_cell_4/BiasAdd_1:output:0'forward_gru_1/gru_cell_4/Const:output:03forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split«
forward_gru_1/gru_cell_4/addAddV2'forward_gru_1/gru_cell_4/split:output:0)forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru_1/gru_cell_4/SigmoidSigmoid forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
forward_gru_1/gru_cell_4/add_1AddV2'forward_gru_1/gru_cell_4/split:output:1)forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid"forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
forward_gru_1/gru_cell_4/mulMul&forward_gru_1/gru_cell_4/Sigmoid_1:y:0)forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
forward_gru_1/gru_cell_4/add_2AddV2'forward_gru_1/gru_cell_4/split:output:2 forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru_1/gru_cell_4/ReluRelu"forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru_1/gru_cell_4/mul_1Mul$forward_gru_1/gru_cell_4/Sigmoid:y:0forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
forward_gru_1/gru_cell_4/subSub'forward_gru_1/gru_cell_4/sub/x:output:0$forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
forward_gru_1/gru_cell_4/mul_2Mul forward_gru_1/gru_cell_4/sub:z:0+forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
forward_gru_1/gru_cell_4/add_3AddV2"forward_gru_1/gru_cell_4/mul_1:z:0"forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
+forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   â
forward_gru_1/TensorArrayV2_1TensorListReserve4forward_gru_1/TensorArrayV2_1/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
forward_gru_1/whileWhile)forward_gru_1/while/loop_counter:output:0/forward_gru_1/while/maximum_iterations:output:0forward_gru_1/time:output:0&forward_gru_1/TensorArrayV2_1:handle:0forward_gru_1/zeros:output:0&forward_gru_1/strided_slice_1:output:0Eforward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00forward_gru_1_gru_cell_4_readvariableop_resource7forward_gru_1_gru_cell_4_matmul_readvariableop_resource9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
forward_gru_1_while_body_42286**
cond"R 
forward_gru_1_while_cond_42285*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
>forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ì
0forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru_1/while:output:3Gforward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0v
#forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
forward_gru_1/strided_slice_3StridedSlice9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,forward_gru_1/strided_slice_3/stack:output:0.forward_gru_1/strided_slice_3/stack_1:output:0.forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masks
forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          À
forward_gru_1/transpose_1	Transpose9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    J
backward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:l
"backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru_1/strided_sliceStridedSlicebackward_gru_1/Shape:output:0+backward_gru_1/strided_slice/stack:output:0-backward_gru_1/strided_slice/stack_1:output:0-backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
 
backward_gru_1/zeros/packedPack%backward_gru_1/strided_slice:output:0&backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru_1/zerosFill$backward_gru_1/zeros/packed:output:0#backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru_1/transpose	Transposeinputs&backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
backward_gru_1/Shape_1Shapebackward_gru_1/transpose:y:0*
T0*
_output_shapes
:n
$backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
backward_gru_1/strided_slice_1StridedSlicebackward_gru_1/Shape_1:output:0-backward_gru_1/strided_slice_1/stack:output:0/backward_gru_1/strided_slice_1/stack_1:output:0/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
backward_gru_1/TensorArrayV2TensorListReserve3backward_gru_1/TensorArrayV2/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¡
backward_gru_1/ReverseV2	ReverseV2backward_gru_1/transpose:y:0&backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!backward_gru_1/ReverseV2:output:0Mbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
backward_gru_1/strided_slice_2StridedSlicebackward_gru_1/transpose:y:0-backward_gru_1/strided_slice_2/stack:output:0/backward_gru_1/strided_slice_2/stack_1:output:0/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOp1backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0
!backward_gru_1/gru_cell_5/unstackUnpack0backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¨
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¾
 backward_gru_1/gru_cell_5/MatMulMatMul'backward_gru_1/strided_slice_2:output:07backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!backward_gru_1/gru_cell_5/BiasAddBiasAdd*backward_gru_1/gru_cell_5/MatMul:product:0*backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
)backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
backward_gru_1/gru_cell_5/splitSplit2backward_gru_1/gru_cell_5/split/split_dim:output:0*backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¬
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¸
"backward_gru_1/gru_cell_5/MatMul_1MatMulbackward_gru_1/zeros:output:09backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd,backward_gru_1/gru_cell_5/MatMul_1:product:0*backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿv
+backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ®
!backward_gru_1/gru_cell_5/split_1SplitV,backward_gru_1/gru_cell_5/BiasAdd_1:output:0(backward_gru_1/gru_cell_5/Const:output:04backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split®
backward_gru_1/gru_cell_5/addAddV2(backward_gru_1/gru_cell_5/split:output:0*backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru_1/gru_cell_5/SigmoidSigmoid!backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
backward_gru_1/gru_cell_5/add_1AddV2(backward_gru_1/gru_cell_5/split:output:1*backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid#backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
backward_gru_1/gru_cell_5/mulMul'backward_gru_1/gru_cell_5/Sigmoid_1:y:0*backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
backward_gru_1/gru_cell_5/add_2AddV2(backward_gru_1/gru_cell_5/split:output:2!backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru_1/gru_cell_5/ReluRelu#backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru_1/gru_cell_5/mul_1Mul%backward_gru_1/gru_cell_5/Sigmoid:y:0backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
backward_gru_1/gru_cell_5/subSub(backward_gru_1/gru_cell_5/sub/x:output:0%backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
backward_gru_1/gru_cell_5/mul_2Mul!backward_gru_1/gru_cell_5/sub:z:0,backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
backward_gru_1/gru_cell_5/add_3AddV2#backward_gru_1/gru_cell_5/mul_1:z:0#backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
,backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   å
backward_gru_1/TensorArrayV2_1TensorListReserve5backward_gru_1/TensorArrayV2_1/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü
backward_gru_1/whileWhile*backward_gru_1/while/loop_counter:output:00backward_gru_1/while/maximum_iterations:output:0backward_gru_1/time:output:0'backward_gru_1/TensorArrayV2_1:handle:0backward_gru_1/zeros:output:0'backward_gru_1/strided_slice_1:output:0Fbackward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01backward_gru_1_gru_cell_5_readvariableop_resource8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
backward_gru_1_while_body_42437*+
cond#R!
backward_gru_1_while_cond_42436*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
?backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ï
1backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru_1/while:output:3Hbackward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0w
$backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
backward_gru_1/strided_slice_3StridedSlice:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0-backward_gru_1/strided_slice_3/stack:output:0/backward_gru_1/strided_slice_3/stack_1:output:0/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskt
backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
backward_gru_1/transpose_1	Transpose:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0(backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :´
concatConcatV2&forward_gru_1/strided_slice_3:output:0'backward_gru_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ë
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp0^backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2^backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp)^backward_gru_1/gru_cell_5/ReadVariableOp^backward_gru_1/whileS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp/^forward_gru_1/gru_cell_4/MatMul/ReadVariableOp1^forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp(^forward_gru_1/gru_cell_4/ReadVariableOp^forward_gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2b
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2f
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2T
(backward_gru_1/gru_cell_5/ReadVariableOp(backward_gru_1/gru_cell_5/ReadVariableOp2,
backward_gru_1/whilebackward_gru_1/while2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2`
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2d
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2R
'forward_gru_1/gru_cell_4/ReadVariableOp'forward_gru_1/gru_cell_4/ReadVariableOp2*
forward_gru_1/whileforward_gru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹<
ø
while_body_46141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_5_readvariableop_resource_0:C
1while_gru_cell_5_matmul_readvariableop_resource_0:E
3while_gru_cell_5_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_5_readvariableop_resource:A
/while_gru_cell_5_matmul_readvariableop_resource:C
1while_gru_cell_5_matmul_1_readvariableop_resource:
¢&while/gru_cell_5/MatMul/ReadVariableOp¢(while/gru_cell_5/MatMul_1/ReadVariableOp¢while/gru_cell_5/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
×,
þ
G__inference_sequential_1_layer_call_and_return_conditional_losses_43188
bidirectional_1_input'
bidirectional_1_43152:'
bidirectional_1_43154:'
bidirectional_1_43156:
'
bidirectional_1_43158:'
bidirectional_1_43160:'
bidirectional_1_43162:

dense_2_43165:
dense_2_43167:
dense_3_43170:
dense_3_43172:
identity¢'bidirectional_1/StatefulPartitionedCall¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallü
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCallbidirectional_1_inputbidirectional_1_43152bidirectional_1_43154bidirectional_1_43156bidirectional_1_43158bidirectional_1_43160bidirectional_1_43162*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42981
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_43165dense_2_43167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42565
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_43170dense_3_43172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42582§
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_43154*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_43160*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOp(^bidirectional_1/StatefulPartitionedCallS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:b ^
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namebidirectional_1_input
ß
¡
while_body_40793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_4_40815_0:*
while_gru_cell_4_40817_0:*
while_gru_cell_4_40819_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_4_40815:(
while_gru_cell_4_40817:(
while_gru_cell_4_40819:
¢(while/gru_cell_4/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_40815_0while_gru_cell_4_40817_0while_gru_cell_4_40819_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40780Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_4_40815while_gru_cell_4_40815_0"2
while_gru_cell_4_40817while_gru_cell_4_40817_0"2
while_gru_cell_4_40819while_gru_cell_4_40819_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_45452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45452___redundant_placeholder03
/while_while_cond_45452___redundant_placeholder13
/while_while_cond_45452___redundant_placeholder23
/while_while_cond_45452___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
÷(
ª
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_46798

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
¦
»
-__inference_forward_gru_1_layer_call_fn_45389

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_42158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_41669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_41669___redundant_placeholder03
/while_while_cond_41669___redundant_placeholder13
/while_while_cond_41669___redundant_placeholder23
/while_while_cond_41669___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
ëÖ

J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42981

inputsB
0forward_gru_1_gru_cell_4_readvariableop_resource:I
7forward_gru_1_gru_cell_4_matmul_readvariableop_resource:K
9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
C
1backward_gru_1_gru_cell_5_readvariableop_resource:J
8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:L
:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:

identity¢/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢(backward_gru_1/gru_cell_5/ReadVariableOp¢backward_gru_1/while¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢'forward_gru_1/gru_cell_4/ReadVariableOp¢forward_gru_1/whileI
forward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:k
!forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru_1/strided_sliceStridedSliceforward_gru_1/Shape:output:0*forward_gru_1/strided_slice/stack:output:0,forward_gru_1/strided_slice/stack_1:output:0,forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru_1/zeros/packedPack$forward_gru_1/strided_slice:output:0%forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru_1/zerosFill#forward_gru_1/zeros/packed:output:0"forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru_1/transpose	Transposeinputs%forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
forward_gru_1/Shape_1Shapeforward_gru_1/transpose:y:0*
T0*
_output_shapes
:m
#forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
forward_gru_1/strided_slice_1StridedSliceforward_gru_1/Shape_1:output:0,forward_gru_1/strided_slice_1/stack:output:0.forward_gru_1/strided_slice_1/stack_1:output:0.forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
forward_gru_1/TensorArrayV2TensorListReserve2forward_gru_1/TensorArrayV2/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
5forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru_1/transpose:y:0Lforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
forward_gru_1/strided_slice_2StridedSliceforward_gru_1/transpose:y:0,forward_gru_1/strided_slice_2/stack:output:0.forward_gru_1/strided_slice_2/stack_1:output:0.forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp0forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0
 forward_gru_1/gru_cell_4/unstackUnpack/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0»
forward_gru_1/gru_cell_4/MatMulMatMul&forward_gru_1/strided_slice_2:output:06forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 forward_gru_1/gru_cell_4/BiasAddBiasAdd)forward_gru_1/gru_cell_4/MatMul:product:0)forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
forward_gru_1/gru_cell_4/splitSplit1forward_gru_1/gru_cell_4/split/split_dim:output:0)forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitª
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0µ
!forward_gru_1/gru_cell_4/MatMul_1MatMulforward_gru_1/zeros:output:08forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
"forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd+forward_gru_1/gru_cell_4/MatMul_1:product:0)forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿu
*forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
 forward_gru_1/gru_cell_4/split_1SplitV+forward_gru_1/gru_cell_4/BiasAdd_1:output:0'forward_gru_1/gru_cell_4/Const:output:03forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split«
forward_gru_1/gru_cell_4/addAddV2'forward_gru_1/gru_cell_4/split:output:0)forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru_1/gru_cell_4/SigmoidSigmoid forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
forward_gru_1/gru_cell_4/add_1AddV2'forward_gru_1/gru_cell_4/split:output:1)forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid"forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
forward_gru_1/gru_cell_4/mulMul&forward_gru_1/gru_cell_4/Sigmoid_1:y:0)forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
forward_gru_1/gru_cell_4/add_2AddV2'forward_gru_1/gru_cell_4/split:output:2 forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru_1/gru_cell_4/ReluRelu"forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru_1/gru_cell_4/mul_1Mul$forward_gru_1/gru_cell_4/Sigmoid:y:0forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
forward_gru_1/gru_cell_4/subSub'forward_gru_1/gru_cell_4/sub/x:output:0$forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
forward_gru_1/gru_cell_4/mul_2Mul forward_gru_1/gru_cell_4/sub:z:0+forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
forward_gru_1/gru_cell_4/add_3AddV2"forward_gru_1/gru_cell_4/mul_1:z:0"forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
+forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   â
forward_gru_1/TensorArrayV2_1TensorListReserve4forward_gru_1/TensorArrayV2_1/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
forward_gru_1/whileWhile)forward_gru_1/while/loop_counter:output:0/forward_gru_1/while/maximum_iterations:output:0forward_gru_1/time:output:0&forward_gru_1/TensorArrayV2_1:handle:0forward_gru_1/zeros:output:0&forward_gru_1/strided_slice_1:output:0Eforward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00forward_gru_1_gru_cell_4_readvariableop_resource7forward_gru_1_gru_cell_4_matmul_readvariableop_resource9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
forward_gru_1_while_body_42727**
cond"R 
forward_gru_1_while_cond_42726*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
>forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ì
0forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru_1/while:output:3Gforward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0v
#forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
forward_gru_1/strided_slice_3StridedSlice9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,forward_gru_1/strided_slice_3/stack:output:0.forward_gru_1/strided_slice_3/stack_1:output:0.forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masks
forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          À
forward_gru_1/transpose_1	Transpose9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    J
backward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:l
"backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru_1/strided_sliceStridedSlicebackward_gru_1/Shape:output:0+backward_gru_1/strided_slice/stack:output:0-backward_gru_1/strided_slice/stack_1:output:0-backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
 
backward_gru_1/zeros/packedPack%backward_gru_1/strided_slice:output:0&backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru_1/zerosFill$backward_gru_1/zeros/packed:output:0#backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru_1/transpose	Transposeinputs&backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
backward_gru_1/Shape_1Shapebackward_gru_1/transpose:y:0*
T0*
_output_shapes
:n
$backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
backward_gru_1/strided_slice_1StridedSlicebackward_gru_1/Shape_1:output:0-backward_gru_1/strided_slice_1/stack:output:0/backward_gru_1/strided_slice_1/stack_1:output:0/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
backward_gru_1/TensorArrayV2TensorListReserve3backward_gru_1/TensorArrayV2/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¡
backward_gru_1/ReverseV2	ReverseV2backward_gru_1/transpose:y:0&backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!backward_gru_1/ReverseV2:output:0Mbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
backward_gru_1/strided_slice_2StridedSlicebackward_gru_1/transpose:y:0-backward_gru_1/strided_slice_2/stack:output:0/backward_gru_1/strided_slice_2/stack_1:output:0/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOp1backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0
!backward_gru_1/gru_cell_5/unstackUnpack0backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¨
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¾
 backward_gru_1/gru_cell_5/MatMulMatMul'backward_gru_1/strided_slice_2:output:07backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!backward_gru_1/gru_cell_5/BiasAddBiasAdd*backward_gru_1/gru_cell_5/MatMul:product:0*backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
)backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
backward_gru_1/gru_cell_5/splitSplit2backward_gru_1/gru_cell_5/split/split_dim:output:0*backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¬
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¸
"backward_gru_1/gru_cell_5/MatMul_1MatMulbackward_gru_1/zeros:output:09backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd,backward_gru_1/gru_cell_5/MatMul_1:product:0*backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿv
+backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ®
!backward_gru_1/gru_cell_5/split_1SplitV,backward_gru_1/gru_cell_5/BiasAdd_1:output:0(backward_gru_1/gru_cell_5/Const:output:04backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split®
backward_gru_1/gru_cell_5/addAddV2(backward_gru_1/gru_cell_5/split:output:0*backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru_1/gru_cell_5/SigmoidSigmoid!backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
backward_gru_1/gru_cell_5/add_1AddV2(backward_gru_1/gru_cell_5/split:output:1*backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid#backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
backward_gru_1/gru_cell_5/mulMul'backward_gru_1/gru_cell_5/Sigmoid_1:y:0*backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
backward_gru_1/gru_cell_5/add_2AddV2(backward_gru_1/gru_cell_5/split:output:2!backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru_1/gru_cell_5/ReluRelu#backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru_1/gru_cell_5/mul_1Mul%backward_gru_1/gru_cell_5/Sigmoid:y:0backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
backward_gru_1/gru_cell_5/subSub(backward_gru_1/gru_cell_5/sub/x:output:0%backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
backward_gru_1/gru_cell_5/mul_2Mul!backward_gru_1/gru_cell_5/sub:z:0,backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
backward_gru_1/gru_cell_5/add_3AddV2#backward_gru_1/gru_cell_5/mul_1:z:0#backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
,backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   å
backward_gru_1/TensorArrayV2_1TensorListReserve5backward_gru_1/TensorArrayV2_1/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü
backward_gru_1/whileWhile*backward_gru_1/while/loop_counter:output:00backward_gru_1/while/maximum_iterations:output:0backward_gru_1/time:output:0'backward_gru_1/TensorArrayV2_1:handle:0backward_gru_1/zeros:output:0'backward_gru_1/strided_slice_1:output:0Fbackward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01backward_gru_1_gru_cell_5_readvariableop_resource8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
backward_gru_1_while_body_42878*+
cond#R!
backward_gru_1_while_cond_42877*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
?backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ï
1backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru_1/while:output:3Hbackward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0w
$backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
backward_gru_1/strided_slice_3StridedSlice:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0-backward_gru_1/strided_slice_3/stack:output:0/backward_gru_1/strided_slice_3/stack_1:output:0/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskt
backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
backward_gru_1/transpose_1	Transpose:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0(backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :´
concatConcatV2&forward_gru_1/strided_slice_3:output:0'backward_gru_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ë
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp0^backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2^backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp)^backward_gru_1/gru_cell_5/ReadVariableOp^backward_gru_1/whileS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp/^forward_gru_1/gru_cell_4/MatMul/ReadVariableOp1^forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp(^forward_gru_1/gru_cell_4/ReadVariableOp^forward_gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2b
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2f
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2T
(backward_gru_1/gru_cell_5/ReadVariableOp(backward_gru_1/gru_cell_5/ReadVariableOp2,
backward_gru_1/whilebackward_gru_1/while2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2`
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2d
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2R
'forward_gru_1/gru_cell_4/ReadVariableOp'forward_gru_1/gru_cell_4/ReadVariableOp2*
forward_gru_1/whileforward_gru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
N

forward_gru_1_while_body_450458
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_27
3forward_gru_1_while_forward_gru_1_strided_slice_1_0s
oforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0J
8forward_gru_1_while_gru_cell_4_readvariableop_resource_0:Q
?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:S
Aforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
 
forward_gru_1_while_identity"
forward_gru_1_while_identity_1"
forward_gru_1_while_identity_2"
forward_gru_1_while_identity_3"
forward_gru_1_while_identity_45
1forward_gru_1_while_forward_gru_1_strided_slice_1q
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensorH
6forward_gru_1_while_gru_cell_4_readvariableop_resource:O
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:Q
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢-forward_gru_1/while/gru_cell_4/ReadVariableOp
Eforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
7forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0forward_gru_1_while_placeholderNforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¦
-forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOp8forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
&forward_gru_1/while/gru_cell_4/unstackUnpack5forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num´
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ß
%forward_gru_1/while/gru_cell_4/MatMulMatMul>forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&forward_gru_1/while/gru_cell_4/BiasAddBiasAdd/forward_gru_1/while/gru_cell_4/MatMul:product:0/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$forward_gru_1/while/gru_cell_4/splitSplit7forward_gru_1/while/gru_cell_4/split/split_dim:output:0/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¸
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Æ
'forward_gru_1/while/gru_cell_4/MatMul_1MatMul!forward_gru_1_while_placeholder_2>forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAdd1forward_gru_1/while/gru_cell_4/MatMul_1:product:0/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ{
0forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&forward_gru_1/while/gru_cell_4/split_1SplitV1forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0-forward_gru_1/while/gru_cell_4/Const:output:09forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split½
"forward_gru_1/while/gru_cell_4/addAddV2-forward_gru_1/while/gru_cell_4/split:output:0/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru_1/while/gru_cell_4/SigmoidSigmoid&forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
$forward_gru_1/while/gru_cell_4/add_1AddV2-forward_gru_1/while/gru_cell_4/split:output:1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid(forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
"forward_gru_1/while/gru_cell_4/mulMul,forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
$forward_gru_1/while/gru_cell_4/add_2AddV2-forward_gru_1/while/gru_cell_4/split:output:2&forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#forward_gru_1/while/gru_cell_4/ReluRelu(forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
$forward_gru_1/while/gru_cell_4/mul_1Mul*forward_gru_1/while/gru_cell_4/Sigmoid:y:0!forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
$forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"forward_gru_1/while/gru_cell_4/subSub-forward_gru_1/while/gru_cell_4/sub/x:output:0*forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
$forward_gru_1/while/gru_cell_4/mul_2Mul&forward_gru_1/while/gru_cell_4/sub:z:01forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
$forward_gru_1/while/gru_cell_4/add_3AddV2(forward_gru_1/while/gru_cell_4/mul_1:z:0(forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
8forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!forward_gru_1_while_placeholder_1forward_gru_1_while_placeholder(forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/addAddV2forward_gru_1_while_placeholder"forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ]
forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/add_1AddV24forward_gru_1_while_forward_gru_1_while_loop_counter$forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_gru_1/while/IdentityIdentityforward_gru_1/while/add_1:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: ¢
forward_gru_1/while/Identity_1Identity:forward_gru_1_while_forward_gru_1_while_maximum_iterations^forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
forward_gru_1/while/Identity_2Identityforward_gru_1/while/add:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ã
forward_gru_1/while/Identity_3IdentityHforward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¡
forward_gru_1/while/Identity_4Identity(forward_gru_1/while/gru_cell_4/add_3:z:0^forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
forward_gru_1/while/NoOpNoOp5^forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp7^forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp.^forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1forward_gru_1_while_forward_gru_1_strided_slice_13forward_gru_1_while_forward_gru_1_strided_slice_1_0"
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"r
6forward_gru_1_while_gru_cell_4_readvariableop_resource8forward_gru_1_while_gru_cell_4_readvariableop_resource_0"E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0"I
forward_gru_1_while_identity_1'forward_gru_1/while/Identity_1:output:0"I
forward_gru_1_while_identity_2'forward_gru_1/while/Identity_2:output:0"I
forward_gru_1_while_identity_3'forward_gru_1/while/Identity_3:output:0"I
forward_gru_1_while_identity_4'forward_gru_1/while/Identity_4:output:0"à
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensoroforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2l
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2p
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2^
-forward_gru_1/while/gru_cell_4/ReadVariableOp-forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
)
«
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_46933

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
ï(
¨
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40935

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates

½
-__inference_forward_gru_1_layer_call_fn_45356
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_40863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Û?
Ô
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_41057

inputs"
gru_cell_4_40975:"
gru_cell_4_40977:"
gru_cell_4_40979:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_4/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÀ
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_40975gru_cell_4_40977gru_cell_4_40979*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40935n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_40975gru_cell_4_40977gru_cell_4_40979*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_40987*
condR
while_cond_40986*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¢
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_4_40977*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ç
NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_2_layer_call_and_return_conditional_losses_42565

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ò
/bidirectional_1_backward_gru_1_while_cond_43802Z
Vbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_loop_counter`
\bidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_maximum_iterations4
0bidirectional_1_backward_gru_1_while_placeholder6
2bidirectional_1_backward_gru_1_while_placeholder_16
2bidirectional_1_backward_gru_1_while_placeholder_2\
Xbidirectional_1_backward_gru_1_while_less_bidirectional_1_backward_gru_1_strided_slice_1q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43802___redundant_placeholder0q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43802___redundant_placeholder1q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43802___redundant_placeholder2q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43802___redundant_placeholder31
-bidirectional_1_backward_gru_1_while_identity
Þ
)bidirectional_1/backward_gru_1/while/LessLess0bidirectional_1_backward_gru_1_while_placeholderXbidirectional_1_backward_gru_1_while_less_bidirectional_1_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: 
-bidirectional_1/backward_gru_1/while/IdentityIdentity-bidirectional_1/backward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "g
-bidirectional_1_backward_gru_1_while_identity6bidirectional_1/backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
N

forward_gru_1_while_body_422868
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_27
3forward_gru_1_while_forward_gru_1_strided_slice_1_0s
oforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0J
8forward_gru_1_while_gru_cell_4_readvariableop_resource_0:Q
?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:S
Aforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
 
forward_gru_1_while_identity"
forward_gru_1_while_identity_1"
forward_gru_1_while_identity_2"
forward_gru_1_while_identity_3"
forward_gru_1_while_identity_45
1forward_gru_1_while_forward_gru_1_strided_slice_1q
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensorH
6forward_gru_1_while_gru_cell_4_readvariableop_resource:O
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:Q
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢-forward_gru_1/while/gru_cell_4/ReadVariableOp
Eforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
7forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0forward_gru_1_while_placeholderNforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¦
-forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOp8forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
&forward_gru_1/while/gru_cell_4/unstackUnpack5forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num´
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ß
%forward_gru_1/while/gru_cell_4/MatMulMatMul>forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&forward_gru_1/while/gru_cell_4/BiasAddBiasAdd/forward_gru_1/while/gru_cell_4/MatMul:product:0/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$forward_gru_1/while/gru_cell_4/splitSplit7forward_gru_1/while/gru_cell_4/split/split_dim:output:0/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¸
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Æ
'forward_gru_1/while/gru_cell_4/MatMul_1MatMul!forward_gru_1_while_placeholder_2>forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAdd1forward_gru_1/while/gru_cell_4/MatMul_1:product:0/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ{
0forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&forward_gru_1/while/gru_cell_4/split_1SplitV1forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0-forward_gru_1/while/gru_cell_4/Const:output:09forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split½
"forward_gru_1/while/gru_cell_4/addAddV2-forward_gru_1/while/gru_cell_4/split:output:0/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru_1/while/gru_cell_4/SigmoidSigmoid&forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
$forward_gru_1/while/gru_cell_4/add_1AddV2-forward_gru_1/while/gru_cell_4/split:output:1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid(forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
"forward_gru_1/while/gru_cell_4/mulMul,forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
$forward_gru_1/while/gru_cell_4/add_2AddV2-forward_gru_1/while/gru_cell_4/split:output:2&forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#forward_gru_1/while/gru_cell_4/ReluRelu(forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
$forward_gru_1/while/gru_cell_4/mul_1Mul*forward_gru_1/while/gru_cell_4/Sigmoid:y:0!forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
$forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"forward_gru_1/while/gru_cell_4/subSub-forward_gru_1/while/gru_cell_4/sub/x:output:0*forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
$forward_gru_1/while/gru_cell_4/mul_2Mul&forward_gru_1/while/gru_cell_4/sub:z:01forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
$forward_gru_1/while/gru_cell_4/add_3AddV2(forward_gru_1/while/gru_cell_4/mul_1:z:0(forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
8forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!forward_gru_1_while_placeholder_1forward_gru_1_while_placeholder(forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/addAddV2forward_gru_1_while_placeholder"forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ]
forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/add_1AddV24forward_gru_1_while_forward_gru_1_while_loop_counter$forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_gru_1/while/IdentityIdentityforward_gru_1/while/add_1:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: ¢
forward_gru_1/while/Identity_1Identity:forward_gru_1_while_forward_gru_1_while_maximum_iterations^forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
forward_gru_1/while/Identity_2Identityforward_gru_1/while/add:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ã
forward_gru_1/while/Identity_3IdentityHforward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¡
forward_gru_1/while/Identity_4Identity(forward_gru_1/while/gru_cell_4/add_3:z:0^forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
forward_gru_1/while/NoOpNoOp5^forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp7^forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp.^forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1forward_gru_1_while_forward_gru_1_strided_slice_13forward_gru_1_while_forward_gru_1_strided_slice_1_0"
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"r
6forward_gru_1_while_gru_cell_4_readvariableop_resource8forward_gru_1_while_gru_cell_4_readvariableop_resource_0"E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0"I
forward_gru_1_while_identity_1'forward_gru_1/while/Identity_1:output:0"I
forward_gru_1_while_identity_2'forward_gru_1/while/Identity_2:output:0"I
forward_gru_1_while_identity_3'forward_gru_1/while/Identity_3:output:0"I
forward_gru_1_while_identity_4'forward_gru_1/while/Identity_4:output:0"à
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensoroforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2l
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2p
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2^
-forward_gru_1/while/gru_cell_4/ReadVariableOp-forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
´	

/__inference_bidirectional_1_layer_call_fn_43976
inputs_0
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_41788o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¬

ý
,__inference_sequential_1_layer_call_fn_43231

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_42601o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_45929
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45929___redundant_placeholder03
/while_while_cond_45929___redundant_placeholder13
/while_while_cond_45929___redundant_placeholder23
/while_while_cond_45929___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
	

/__inference_bidirectional_1_layer_call_fn_44010

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_45770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45770___redundant_placeholder03
/while_while_cond_45770___redundant_placeholder13
/while_while_cond_45770___redundant_placeholder23
/while_while_cond_45770___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
ë

Â
backward_gru_1_while_cond_44559:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_2<
8backward_gru_1_while_less_backward_gru_1_strided_slice_1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44559___redundant_placeholder0Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44559___redundant_placeholder1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44559___redundant_placeholder2Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44559___redundant_placeholder3!
backward_gru_1_while_identity

backward_gru_1/while/LessLess backward_gru_1_while_placeholder8backward_gru_1_while_less_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: i
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:

ò
__inference_loss_fn_1_46989m
[bidirectional_1_backward_gru_1_gru_cell_5_kernel_regularizer_square_readvariableop_resource:
identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpî
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp[bidirectional_1_backward_gru_1_gru_cell_5_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
IdentityIdentityDbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp
áX
ô
__inference__traced_save_47129
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopN
Jsavev2_bidirectional_1_forward_gru_1_gru_cell_4_kernel_read_readvariableopX
Tsavev2_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_read_readvariableopL
Hsavev2_bidirectional_1_forward_gru_1_gru_cell_4_bias_read_readvariableopO
Ksavev2_bidirectional_1_backward_gru_1_gru_cell_5_kernel_read_readvariableopY
Usavev2_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_read_readvariableopM
Isavev2_bidirectional_1_backward_gru_1_gru_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableopU
Qsavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_m_read_readvariableop_
[savev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_m_read_readvariableopS
Osavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_m_read_readvariableopV
Rsavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_m_read_readvariableop`
\savev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_m_read_readvariableopT
Psavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableopU
Qsavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_v_read_readvariableop_
[savev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_v_read_readvariableopS
Osavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_v_read_readvariableopV
Rsavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_v_read_readvariableop`
\savev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_v_read_readvariableopT
Psavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Û
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueúB÷(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH½
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ä
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopJsavev2_bidirectional_1_forward_gru_1_gru_cell_4_kernel_read_readvariableopTsavev2_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_read_readvariableopHsavev2_bidirectional_1_forward_gru_1_gru_cell_4_bias_read_readvariableopKsavev2_bidirectional_1_backward_gru_1_gru_cell_5_kernel_read_readvariableopUsavev2_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_read_readvariableopIsavev2_bidirectional_1_backward_gru_1_gru_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableopQsavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_m_read_readvariableop[savev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_m_read_readvariableopOsavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_m_read_readvariableopRsavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_m_read_readvariableop\savev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_m_read_readvariableopPsavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopQsavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_v_read_readvariableop[savev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_v_read_readvariableopOsavev2_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_v_read_readvariableopRsavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_v_read_readvariableop\savev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_v_read_readvariableopPsavev2_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*¿
_input_shapes­
ª: ::::: : : : : ::
:::
:: : : : ::::::
:::
:::::::
:::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

:
:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:
:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
:$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

::$# 

_output_shapes

:
:$$ 

_output_shapes

::$% 

_output_shapes

::$& 

_output_shapes

:
:$' 

_output_shapes

::(

_output_shapes
: 
Õ
¥
while_cond_41352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_41352___redundant_placeholder03
/while_while_cond_41352___redundant_placeholder13
/while_while_cond_41352___redundant_placeholder23
/while_while_cond_41352___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
¤O
º
backward_gru_1_while_body_45196:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_29
5backward_gru_1_while_backward_gru_1_strided_slice_1_0u
qbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0K
9backward_gru_1_while_gru_cell_5_readvariableop_resource_0:R
@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:T
Bbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
!
backward_gru_1_while_identity#
backward_gru_1_while_identity_1#
backward_gru_1_while_identity_2#
backward_gru_1_while_identity_3#
backward_gru_1_while_identity_47
3backward_gru_1_while_backward_gru_1_strided_slice_1s
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorI
7backward_gru_1_while_gru_cell_5_readvariableop_resource:P
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:R
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢.backward_gru_1/while/gru_cell_5/ReadVariableOp
Fbackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0 backward_gru_1_while_placeholderObackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp9backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
'backward_gru_1/while/gru_cell_5/unstackUnpack6backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¶
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0â
&backward_gru_1/while/gru_cell_5/MatMulMatMul?backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'backward_gru_1/while/gru_cell_5/BiasAddBiasAdd0backward_gru_1/while/gru_cell_5/MatMul:product:00backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
%backward_gru_1/while/gru_cell_5/splitSplit8backward_gru_1/while/gru_cell_5/split/split_dim:output:00backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0É
(backward_gru_1/while/gru_cell_5/MatMul_1MatMul"backward_gru_1_while_placeholder_2?backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAdd2backward_gru_1/while/gru_cell_5/MatMul_1:product:00backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
%backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ|
1backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
'backward_gru_1/while/gru_cell_5/split_1SplitV2backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0.backward_gru_1/while/gru_cell_5/Const:output:0:backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÀ
#backward_gru_1/while/gru_cell_5/addAddV2.backward_gru_1/while/gru_cell_5/split:output:00backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru_1/while/gru_cell_5/SigmoidSigmoid'backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
%backward_gru_1/while/gru_cell_5/add_1AddV2.backward_gru_1/while/gru_cell_5/split:output:10backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid)backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
#backward_gru_1/while/gru_cell_5/mulMul-backward_gru_1/while/gru_cell_5/Sigmoid_1:y:00backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
%backward_gru_1/while/gru_cell_5/add_2AddV2.backward_gru_1/while/gru_cell_5/split:output:2'backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$backward_gru_1/while/gru_cell_5/ReluRelu)backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
%backward_gru_1/while/gru_cell_5/mul_1Mul+backward_gru_1/while/gru_cell_5/Sigmoid:y:0"backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
%backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
#backward_gru_1/while/gru_cell_5/subSub.backward_gru_1/while/gru_cell_5/sub/x:output:0+backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
%backward_gru_1/while/gru_cell_5/mul_2Mul'backward_gru_1/while/gru_cell_5/sub:z:02backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
%backward_gru_1/while/gru_cell_5/add_3AddV2)backward_gru_1/while/gru_cell_5/mul_1:z:0)backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"backward_gru_1_while_placeholder_1 backward_gru_1_while_placeholder)backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru_1/while/addAddV2 backward_gru_1_while_placeholder#backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ^
backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
backward_gru_1/while/add_1AddV26backward_gru_1_while_backward_gru_1_while_loop_counter%backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/add_1:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¦
backward_gru_1/while/Identity_1Identity<backward_gru_1_while_backward_gru_1_while_maximum_iterations^backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
backward_gru_1/while/Identity_2Identitybackward_gru_1/while/add:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: Æ
backward_gru_1/while/Identity_3IdentityIbackward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
backward_gru_1/while/Identity_4Identity)backward_gru_1/while/gru_cell_5/add_3:z:0^backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
backward_gru_1/while/NoOpNoOp6^backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp8^backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp/^backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3backward_gru_1_while_backward_gru_1_strided_slice_15backward_gru_1_while_backward_gru_1_strided_slice_1_0"
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"t
7backward_gru_1_while_gru_cell_5_readvariableop_resource9backward_gru_1_while_gru_cell_5_readvariableop_resource_0"G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0"K
backward_gru_1_while_identity_1(backward_gru_1/while/Identity_1:output:0"K
backward_gru_1_while_identity_2(backward_gru_1/while/Identity_2:output:0"K
backward_gru_1_while_identity_3(backward_gru_1/while/Identity_3:output:0"K
backward_gru_1_while_identity_4(backward_gru_1/while/Identity_4:output:0"ä
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2n
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2r
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2`
.backward_gru_1/while/gru_cell_5/ReadVariableOp.backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_41156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_41156___redundant_placeholder03
/while_while_cond_41156___redundant_placeholder13
/while_while_cond_41156___redundant_placeholder23
/while_while_cond_41156___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
N

forward_gru_1_while_body_447278
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_27
3forward_gru_1_while_forward_gru_1_strided_slice_1_0s
oforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0J
8forward_gru_1_while_gru_cell_4_readvariableop_resource_0:Q
?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:S
Aforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
 
forward_gru_1_while_identity"
forward_gru_1_while_identity_1"
forward_gru_1_while_identity_2"
forward_gru_1_while_identity_3"
forward_gru_1_while_identity_45
1forward_gru_1_while_forward_gru_1_strided_slice_1q
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensorH
6forward_gru_1_while_gru_cell_4_readvariableop_resource:O
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:Q
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢-forward_gru_1/while/gru_cell_4/ReadVariableOp
Eforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
7forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0forward_gru_1_while_placeholderNforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¦
-forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOp8forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
&forward_gru_1/while/gru_cell_4/unstackUnpack5forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num´
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ß
%forward_gru_1/while/gru_cell_4/MatMulMatMul>forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&forward_gru_1/while/gru_cell_4/BiasAddBiasAdd/forward_gru_1/while/gru_cell_4/MatMul:product:0/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$forward_gru_1/while/gru_cell_4/splitSplit7forward_gru_1/while/gru_cell_4/split/split_dim:output:0/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¸
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Æ
'forward_gru_1/while/gru_cell_4/MatMul_1MatMul!forward_gru_1_while_placeholder_2>forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAdd1forward_gru_1/while/gru_cell_4/MatMul_1:product:0/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ{
0forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&forward_gru_1/while/gru_cell_4/split_1SplitV1forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0-forward_gru_1/while/gru_cell_4/Const:output:09forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split½
"forward_gru_1/while/gru_cell_4/addAddV2-forward_gru_1/while/gru_cell_4/split:output:0/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru_1/while/gru_cell_4/SigmoidSigmoid&forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
$forward_gru_1/while/gru_cell_4/add_1AddV2-forward_gru_1/while/gru_cell_4/split:output:1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid(forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
"forward_gru_1/while/gru_cell_4/mulMul,forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
$forward_gru_1/while/gru_cell_4/add_2AddV2-forward_gru_1/while/gru_cell_4/split:output:2&forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#forward_gru_1/while/gru_cell_4/ReluRelu(forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
$forward_gru_1/while/gru_cell_4/mul_1Mul*forward_gru_1/while/gru_cell_4/Sigmoid:y:0!forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
$forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"forward_gru_1/while/gru_cell_4/subSub-forward_gru_1/while/gru_cell_4/sub/x:output:0*forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
$forward_gru_1/while/gru_cell_4/mul_2Mul&forward_gru_1/while/gru_cell_4/sub:z:01forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
$forward_gru_1/while/gru_cell_4/add_3AddV2(forward_gru_1/while/gru_cell_4/mul_1:z:0(forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
8forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!forward_gru_1_while_placeholder_1forward_gru_1_while_placeholder(forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/addAddV2forward_gru_1_while_placeholder"forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ]
forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/add_1AddV24forward_gru_1_while_forward_gru_1_while_loop_counter$forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_gru_1/while/IdentityIdentityforward_gru_1/while/add_1:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: ¢
forward_gru_1/while/Identity_1Identity:forward_gru_1_while_forward_gru_1_while_maximum_iterations^forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
forward_gru_1/while/Identity_2Identityforward_gru_1/while/add:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ã
forward_gru_1/while/Identity_3IdentityHforward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¡
forward_gru_1/while/Identity_4Identity(forward_gru_1/while/gru_cell_4/add_3:z:0^forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
forward_gru_1/while/NoOpNoOp5^forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp7^forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp.^forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1forward_gru_1_while_forward_gru_1_strided_slice_13forward_gru_1_while_forward_gru_1_strided_slice_1_0"
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"r
6forward_gru_1_while_gru_cell_4_readvariableop_resource8forward_gru_1_while_gru_cell_4_readvariableop_resource_0"E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0"I
forward_gru_1_while_identity_1'forward_gru_1/while/Identity_1:output:0"I
forward_gru_1_while_identity_2'forward_gru_1/while/Identity_2:output:0"I
forward_gru_1_while_identity_3'forward_gru_1/while/Identity_3:output:0"I
forward_gru_1_while_identity_4'forward_gru_1/while/Identity_4:output:0"à
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensoroforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2l
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2p
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2^
-forward_gru_1/while/gru_cell_4/ReadVariableOp-forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
¨
¼
.__inference_backward_gru_1_layer_call_fn_46064

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41765o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ùs
¤
<sequential_1_bidirectional_1_backward_gru_1_while_body_40599t
psequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_loop_counterz
vsequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_maximum_iterationsA
=sequential_1_bidirectional_1_backward_gru_1_while_placeholderC
?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_1C
?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_2s
osequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_strided_slice_1_0°
«sequential_1_bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0h
Vsequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0:o
]sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:q
_sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
>
:sequential_1_bidirectional_1_backward_gru_1_while_identity@
<sequential_1_bidirectional_1_backward_gru_1_while_identity_1@
<sequential_1_bidirectional_1_backward_gru_1_while_identity_2@
<sequential_1_bidirectional_1_backward_gru_1_while_identity_3@
<sequential_1_bidirectional_1_backward_gru_1_while_identity_4q
msequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_strided_slice_1®
©sequential_1_bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensorf
Tsequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource:m
[sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:o
]sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢Rsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢Tsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp´
csequential_1/bidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Usequential_1/bidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem«sequential_1_bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0=sequential_1_bidirectional_1_backward_gru_1_while_placeholderlsequential_1/bidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0â
Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOpVsequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
Dsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/unstackUnpackSsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numð
Rsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp]sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0¹
Csequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMulMatMul\sequential_1/bidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Zsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAddBiasAddMsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul:product:0Msequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/splitSplitUsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split/split_dim:output:0Msequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitô
Tsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp_sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0 
Esequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1MatMul?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_2\sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
Fsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAddOsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1:product:0Msequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Nsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
Dsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split_1SplitVOsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/Const:output:0Wsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
@sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/addAddV2Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:0Msequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ç
Dsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/SigmoidSigmoidDsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_1AddV2Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:1Msequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ë
Fsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid_1SigmoidFsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/mulMulJsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid_1:y:0Msequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_2AddV2Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:2Dsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
Asequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReluReluFsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/mul_1MulHsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid:y:0?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/subSubKsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/sub/x:output:0Hsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/mul_2MulDsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/sub:z:0Osequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_3AddV2Fsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/mul_1:z:0Fsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
Vsequential_1/bidirectional_1/backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem?sequential_1_bidirectional_1_backward_gru_1_while_placeholder_1=sequential_1_bidirectional_1_backward_gru_1_while_placeholderFsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒy
7sequential_1/bidirectional_1/backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :à
5sequential_1/bidirectional_1/backward_gru_1/while/addAddV2=sequential_1_bidirectional_1_backward_gru_1_while_placeholder@sequential_1/bidirectional_1/backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: {
9sequential_1/bidirectional_1/backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
7sequential_1/bidirectional_1/backward_gru_1/while/add_1AddV2psequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_loop_counterBsequential_1/bidirectional_1/backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: Ý
:sequential_1/bidirectional_1/backward_gru_1/while/IdentityIdentity;sequential_1/bidirectional_1/backward_gru_1/while/add_1:z:07^sequential_1/bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
<sequential_1/bidirectional_1/backward_gru_1/while/Identity_1Identityvsequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_while_maximum_iterations7^sequential_1/bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: Ý
<sequential_1/bidirectional_1/backward_gru_1/while/Identity_2Identity9sequential_1/bidirectional_1/backward_gru_1/while/add:z:07^sequential_1/bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
<sequential_1/bidirectional_1/backward_gru_1/while/Identity_3Identityfsequential_1/bidirectional_1/backward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^sequential_1/bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒû
<sequential_1/bidirectional_1/backward_gru_1/while/Identity_4IdentityFsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/add_3:z:07^sequential_1/bidirectional_1/backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
6sequential_1/bidirectional_1/backward_gru_1/while/NoOpNoOpS^sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpU^sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpL^sequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "À
]sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"¼
[sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource]sequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"®
Tsequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resourceVsequential_1_bidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0"
:sequential_1_bidirectional_1_backward_gru_1_while_identityCsequential_1/bidirectional_1/backward_gru_1/while/Identity:output:0"
<sequential_1_bidirectional_1_backward_gru_1_while_identity_1Esequential_1/bidirectional_1/backward_gru_1/while/Identity_1:output:0"
<sequential_1_bidirectional_1_backward_gru_1_while_identity_2Esequential_1/bidirectional_1/backward_gru_1/while/Identity_2:output:0"
<sequential_1_bidirectional_1_backward_gru_1_while_identity_3Esequential_1/bidirectional_1/backward_gru_1/while/Identity_3:output:0"
<sequential_1_bidirectional_1_backward_gru_1_while_identity_4Esequential_1/bidirectional_1/backward_gru_1/while/Identity_4:output:0"à
msequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_strided_slice_1osequential_1_bidirectional_1_backward_gru_1_while_sequential_1_bidirectional_1_backward_gru_1_strided_slice_1_0"Ú
©sequential_1_bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor«sequential_1_bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2¨
Rsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpRsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2¬
Tsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpTsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2
Ksequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOpKsequential_1/bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
N

forward_gru_1_while_body_440918
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_27
3forward_gru_1_while_forward_gru_1_strided_slice_1_0s
oforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0J
8forward_gru_1_while_gru_cell_4_readvariableop_resource_0:Q
?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:S
Aforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
 
forward_gru_1_while_identity"
forward_gru_1_while_identity_1"
forward_gru_1_while_identity_2"
forward_gru_1_while_identity_3"
forward_gru_1_while_identity_45
1forward_gru_1_while_forward_gru_1_strided_slice_1q
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensorH
6forward_gru_1_while_gru_cell_4_readvariableop_resource:O
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:Q
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢-forward_gru_1/while/gru_cell_4/ReadVariableOp
Eforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿõ
7forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0forward_gru_1_while_placeholderNforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¦
-forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOp8forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
&forward_gru_1/while/gru_cell_4/unstackUnpack5forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num´
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ß
%forward_gru_1/while/gru_cell_4/MatMulMatMul>forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&forward_gru_1/while/gru_cell_4/BiasAddBiasAdd/forward_gru_1/while/gru_cell_4/MatMul:product:0/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$forward_gru_1/while/gru_cell_4/splitSplit7forward_gru_1/while/gru_cell_4/split/split_dim:output:0/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¸
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Æ
'forward_gru_1/while/gru_cell_4/MatMul_1MatMul!forward_gru_1_while_placeholder_2>forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAdd1forward_gru_1/while/gru_cell_4/MatMul_1:product:0/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ{
0forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&forward_gru_1/while/gru_cell_4/split_1SplitV1forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0-forward_gru_1/while/gru_cell_4/Const:output:09forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split½
"forward_gru_1/while/gru_cell_4/addAddV2-forward_gru_1/while/gru_cell_4/split:output:0/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru_1/while/gru_cell_4/SigmoidSigmoid&forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
$forward_gru_1/while/gru_cell_4/add_1AddV2-forward_gru_1/while/gru_cell_4/split:output:1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid(forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
"forward_gru_1/while/gru_cell_4/mulMul,forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
$forward_gru_1/while/gru_cell_4/add_2AddV2-forward_gru_1/while/gru_cell_4/split:output:2&forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#forward_gru_1/while/gru_cell_4/ReluRelu(forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
$forward_gru_1/while/gru_cell_4/mul_1Mul*forward_gru_1/while/gru_cell_4/Sigmoid:y:0!forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
$forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"forward_gru_1/while/gru_cell_4/subSub-forward_gru_1/while/gru_cell_4/sub/x:output:0*forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
$forward_gru_1/while/gru_cell_4/mul_2Mul&forward_gru_1/while/gru_cell_4/sub:z:01forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
$forward_gru_1/while/gru_cell_4/add_3AddV2(forward_gru_1/while/gru_cell_4/mul_1:z:0(forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
8forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!forward_gru_1_while_placeholder_1forward_gru_1_while_placeholder(forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/addAddV2forward_gru_1_while_placeholder"forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ]
forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/add_1AddV24forward_gru_1_while_forward_gru_1_while_loop_counter$forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_gru_1/while/IdentityIdentityforward_gru_1/while/add_1:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: ¢
forward_gru_1/while/Identity_1Identity:forward_gru_1_while_forward_gru_1_while_maximum_iterations^forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
forward_gru_1/while/Identity_2Identityforward_gru_1/while/add:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ã
forward_gru_1/while/Identity_3IdentityHforward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¡
forward_gru_1/while/Identity_4Identity(forward_gru_1/while/gru_cell_4/add_3:z:0^forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
forward_gru_1/while/NoOpNoOp5^forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp7^forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp.^forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1forward_gru_1_while_forward_gru_1_strided_slice_13forward_gru_1_while_forward_gru_1_strided_slice_1_0"
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"r
6forward_gru_1_while_gru_cell_4_readvariableop_resource8forward_gru_1_while_gru_cell_4_readvariableop_resource_0"E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0"I
forward_gru_1_while_identity_1'forward_gru_1/while/Identity_1:output:0"I
forward_gru_1_while_identity_2'forward_gru_1/while/Identity_2:output:0"I
forward_gru_1_while_identity_3'forward_gru_1/while/Identity_3:output:0"I
forward_gru_1_while_identity_4'forward_gru_1/while/Identity_4:output:0"à
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensoroforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2l
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2p
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2^
-forward_gru_1/while/gru_cell_4/ReadVariableOp-forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
¬

ý
,__inference_sequential_1_layer_call_fn_43256

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43062o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Y
Ù
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_42158

inputs4
"gru_cell_4_readvariableop_resource:;
)gru_cell_4_matmul_readvariableop_resource:=
+gru_cell_4_matmul_1_readvariableop_resource:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_4/MatMul/ReadVariableOp¢"gru_cell_4/MatMul_1/ReadVariableOp¢gru_cell_4/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_42063*
condR
while_cond_42062*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    »
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷X
Û
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45548
inputs_04
"gru_cell_4_readvariableop_resource:;
)gru_cell_4_matmul_readvariableop_resource:=
+gru_cell_4_matmul_1_readvariableop_resource:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_4/MatMul/ReadVariableOp¢"gru_cell_4/MatMul_1/ReadVariableOp¢gru_cell_4/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_45453*
condR
while_cond_45452*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    »
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ï(
¨
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40780

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
Y
Ù
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_41597

inputs4
"gru_cell_4_readvariableop_resource:;
)gru_cell_4_matmul_readvariableop_resource:=
+gru_cell_4_matmul_1_readvariableop_resource:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_4/MatMul/ReadVariableOp¢"gru_cell_4/MatMul_1/ReadVariableOp¢gru_cell_4/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_41502*
condR
while_cond_41501*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    »
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_41501
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_41501___redundant_placeholder03
/while_while_cond_41501___redundant_placeholder13
/while_while_cond_41501___redundant_placeholder23
/while_while_cond_41501___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
­

Ö
*__inference_gru_cell_5_layer_call_fn_46888

inputs
states_0
unknown:
	unknown_0:
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41299o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
­O
º
backward_gru_1_while_body_44242:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_29
5backward_gru_1_while_backward_gru_1_strided_slice_1_0u
qbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0K
9backward_gru_1_while_gru_cell_5_readvariableop_resource_0:R
@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:T
Bbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
!
backward_gru_1_while_identity#
backward_gru_1_while_identity_1#
backward_gru_1_while_identity_2#
backward_gru_1_while_identity_3#
backward_gru_1_while_identity_47
3backward_gru_1_while_backward_gru_1_strided_slice_1s
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorI
7backward_gru_1_while_gru_cell_5_readvariableop_resource:P
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:R
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢.backward_gru_1/while/gru_cell_5/ReadVariableOp
Fbackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿú
8backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0 backward_gru_1_while_placeholderObackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp9backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
'backward_gru_1/while/gru_cell_5/unstackUnpack6backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¶
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0â
&backward_gru_1/while/gru_cell_5/MatMulMatMul?backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'backward_gru_1/while/gru_cell_5/BiasAddBiasAdd0backward_gru_1/while/gru_cell_5/MatMul:product:00backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
%backward_gru_1/while/gru_cell_5/splitSplit8backward_gru_1/while/gru_cell_5/split/split_dim:output:00backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0É
(backward_gru_1/while/gru_cell_5/MatMul_1MatMul"backward_gru_1_while_placeholder_2?backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAdd2backward_gru_1/while/gru_cell_5/MatMul_1:product:00backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
%backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ|
1backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
'backward_gru_1/while/gru_cell_5/split_1SplitV2backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0.backward_gru_1/while/gru_cell_5/Const:output:0:backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÀ
#backward_gru_1/while/gru_cell_5/addAddV2.backward_gru_1/while/gru_cell_5/split:output:00backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru_1/while/gru_cell_5/SigmoidSigmoid'backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
%backward_gru_1/while/gru_cell_5/add_1AddV2.backward_gru_1/while/gru_cell_5/split:output:10backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid)backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
#backward_gru_1/while/gru_cell_5/mulMul-backward_gru_1/while/gru_cell_5/Sigmoid_1:y:00backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
%backward_gru_1/while/gru_cell_5/add_2AddV2.backward_gru_1/while/gru_cell_5/split:output:2'backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$backward_gru_1/while/gru_cell_5/ReluRelu)backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
%backward_gru_1/while/gru_cell_5/mul_1Mul+backward_gru_1/while/gru_cell_5/Sigmoid:y:0"backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
%backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
#backward_gru_1/while/gru_cell_5/subSub.backward_gru_1/while/gru_cell_5/sub/x:output:0+backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
%backward_gru_1/while/gru_cell_5/mul_2Mul'backward_gru_1/while/gru_cell_5/sub:z:02backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
%backward_gru_1/while/gru_cell_5/add_3AddV2)backward_gru_1/while/gru_cell_5/mul_1:z:0)backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"backward_gru_1_while_placeholder_1 backward_gru_1_while_placeholder)backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru_1/while/addAddV2 backward_gru_1_while_placeholder#backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ^
backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
backward_gru_1/while/add_1AddV26backward_gru_1_while_backward_gru_1_while_loop_counter%backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/add_1:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¦
backward_gru_1/while/Identity_1Identity<backward_gru_1_while_backward_gru_1_while_maximum_iterations^backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
backward_gru_1/while/Identity_2Identitybackward_gru_1/while/add:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: Æ
backward_gru_1/while/Identity_3IdentityIbackward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
backward_gru_1/while/Identity_4Identity)backward_gru_1/while/gru_cell_5/add_3:z:0^backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
backward_gru_1/while/NoOpNoOp6^backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp8^backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp/^backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3backward_gru_1_while_backward_gru_1_strided_slice_15backward_gru_1_while_backward_gru_1_strided_slice_1_0"
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"t
7backward_gru_1_while_gru_cell_5_readvariableop_resource9backward_gru_1_while_gru_cell_5_readvariableop_resource_0"G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0"K
backward_gru_1_while_identity_1(backward_gru_1/while/Identity_1:output:0"K
backward_gru_1_while_identity_2(backward_gru_1/while/Identity_2:output:0"K
backward_gru_1_while_identity_3(backward_gru_1/while/Identity_3:output:0"K
backward_gru_1_while_identity_4(backward_gru_1/while/Identity_4:output:0"ä
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2n
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2r
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2`
.backward_gru_1/while/gru_cell_5/ReadVariableOp.backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
åZ
Ý
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46397
inputs_04
"gru_cell_5_readvariableop_resource:;
)gru_cell_5_matmul_readvariableop_resource:=
+gru_cell_5_matmul_1_readvariableop_resource:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_5/MatMul/ReadVariableOp¢"gru_cell_5/MatMul_1/ReadVariableOp¢gru_cell_5/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46302*
condR
while_cond_46301*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¼
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Y
Ù
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_46025

inputs4
"gru_cell_4_readvariableop_resource:;
)gru_cell_4_matmul_readvariableop_resource:=
+gru_cell_4_matmul_1_readvariableop_resource:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_4/MatMul/ReadVariableOp¢"gru_cell_4/MatMul_1/ReadVariableOp¢gru_cell_4/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_45930*
condR
while_cond_45929*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    »
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

Ö
*__inference_gru_cell_4_layer_call_fn_46739

inputs
states_0
unknown:
	unknown_0:
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40780o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
¤O
º
backward_gru_1_while_body_42437:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_29
5backward_gru_1_while_backward_gru_1_strided_slice_1_0u
qbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0K
9backward_gru_1_while_gru_cell_5_readvariableop_resource_0:R
@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:T
Bbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
!
backward_gru_1_while_identity#
backward_gru_1_while_identity_1#
backward_gru_1_while_identity_2#
backward_gru_1_while_identity_3#
backward_gru_1_while_identity_47
3backward_gru_1_while_backward_gru_1_strided_slice_1s
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorI
7backward_gru_1_while_gru_cell_5_readvariableop_resource:P
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:R
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢.backward_gru_1/while/gru_cell_5/ReadVariableOp
Fbackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0 backward_gru_1_while_placeholderObackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp9backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
'backward_gru_1/while/gru_cell_5/unstackUnpack6backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¶
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0â
&backward_gru_1/while/gru_cell_5/MatMulMatMul?backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'backward_gru_1/while/gru_cell_5/BiasAddBiasAdd0backward_gru_1/while/gru_cell_5/MatMul:product:00backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
%backward_gru_1/while/gru_cell_5/splitSplit8backward_gru_1/while/gru_cell_5/split/split_dim:output:00backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0É
(backward_gru_1/while/gru_cell_5/MatMul_1MatMul"backward_gru_1_while_placeholder_2?backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAdd2backward_gru_1/while/gru_cell_5/MatMul_1:product:00backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
%backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ|
1backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
'backward_gru_1/while/gru_cell_5/split_1SplitV2backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0.backward_gru_1/while/gru_cell_5/Const:output:0:backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÀ
#backward_gru_1/while/gru_cell_5/addAddV2.backward_gru_1/while/gru_cell_5/split:output:00backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru_1/while/gru_cell_5/SigmoidSigmoid'backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
%backward_gru_1/while/gru_cell_5/add_1AddV2.backward_gru_1/while/gru_cell_5/split:output:10backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid)backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
#backward_gru_1/while/gru_cell_5/mulMul-backward_gru_1/while/gru_cell_5/Sigmoid_1:y:00backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
%backward_gru_1/while/gru_cell_5/add_2AddV2.backward_gru_1/while/gru_cell_5/split:output:2'backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$backward_gru_1/while/gru_cell_5/ReluRelu)backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
%backward_gru_1/while/gru_cell_5/mul_1Mul+backward_gru_1/while/gru_cell_5/Sigmoid:y:0"backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
%backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
#backward_gru_1/while/gru_cell_5/subSub.backward_gru_1/while/gru_cell_5/sub/x:output:0+backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
%backward_gru_1/while/gru_cell_5/mul_2Mul'backward_gru_1/while/gru_cell_5/sub:z:02backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
%backward_gru_1/while/gru_cell_5/add_3AddV2)backward_gru_1/while/gru_cell_5/mul_1:z:0)backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"backward_gru_1_while_placeholder_1 backward_gru_1_while_placeholder)backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru_1/while/addAddV2 backward_gru_1_while_placeholder#backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ^
backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
backward_gru_1/while/add_1AddV26backward_gru_1_while_backward_gru_1_while_loop_counter%backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/add_1:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¦
backward_gru_1/while/Identity_1Identity<backward_gru_1_while_backward_gru_1_while_maximum_iterations^backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
backward_gru_1/while/Identity_2Identitybackward_gru_1/while/add:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: Æ
backward_gru_1/while/Identity_3IdentityIbackward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
backward_gru_1/while/Identity_4Identity)backward_gru_1/while/gru_cell_5/add_3:z:0^backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
backward_gru_1/while/NoOpNoOp6^backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp8^backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp/^backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3backward_gru_1_while_backward_gru_1_strided_slice_15backward_gru_1_while_backward_gru_1_strided_slice_1_0"
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"t
7backward_gru_1_while_gru_cell_5_readvariableop_resource9backward_gru_1_while_gru_cell_5_readvariableop_resource_0"G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0"K
backward_gru_1_while_identity_1(backward_gru_1/while/Identity_1:output:0"K
backward_gru_1_while_identity_2(backward_gru_1/while/Identity_2:output:0"K
backward_gru_1_while_identity_3(backward_gru_1/while/Identity_3:output:0"K
backward_gru_1_while_identity_4(backward_gru_1/while/Identity_4:output:0"ä
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2n
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2r
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2`
.backward_gru_1/while/gru_cell_5/ReadVariableOp.backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
­

Ö
*__inference_gru_cell_4_layer_call_fn_46753

inputs
states_0
unknown:
	unknown_0:
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40935o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
[
Û
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46719

inputs4
"gru_cell_5_readvariableop_resource:;
)gru_cell_5_matmul_readvariableop_resource:=
+gru_cell_5_matmul_1_readvariableop_resource:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_5/MatMul/ReadVariableOp¢"gru_cell_5/MatMul_1/ReadVariableOp¢gru_cell_5/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46624*
condR
while_cond_46623*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¼
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â<
ø
while_body_42063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_4_readvariableop_resource_0:C
1while_gru_cell_4_matmul_readvariableop_resource_0:E
3while_gru_cell_4_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_4_readvariableop_resource:A
/while_gru_cell_4_matmul_readvariableop_resource:C
1while_gru_cell_4_matmul_1_readvariableop_resource:
¢&while/gru_cell_4/MatMul/ReadVariableOp¢(while/gru_cell_4/MatMul_1/ReadVariableOp¢while/gru_cell_4/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
N

forward_gru_1_while_body_444098
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_27
3forward_gru_1_while_forward_gru_1_strided_slice_1_0s
oforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0J
8forward_gru_1_while_gru_cell_4_readvariableop_resource_0:Q
?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:S
Aforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
 
forward_gru_1_while_identity"
forward_gru_1_while_identity_1"
forward_gru_1_while_identity_2"
forward_gru_1_while_identity_3"
forward_gru_1_while_identity_45
1forward_gru_1_while_forward_gru_1_strided_slice_1q
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensorH
6forward_gru_1_while_gru_cell_4_readvariableop_resource:O
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:Q
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢-forward_gru_1/while/gru_cell_4/ReadVariableOp
Eforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿõ
7forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0forward_gru_1_while_placeholderNforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¦
-forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOp8forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
&forward_gru_1/while/gru_cell_4/unstackUnpack5forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num´
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ß
%forward_gru_1/while/gru_cell_4/MatMulMatMul>forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&forward_gru_1/while/gru_cell_4/BiasAddBiasAdd/forward_gru_1/while/gru_cell_4/MatMul:product:0/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$forward_gru_1/while/gru_cell_4/splitSplit7forward_gru_1/while/gru_cell_4/split/split_dim:output:0/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¸
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Æ
'forward_gru_1/while/gru_cell_4/MatMul_1MatMul!forward_gru_1_while_placeholder_2>forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAdd1forward_gru_1/while/gru_cell_4/MatMul_1:product:0/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ{
0forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&forward_gru_1/while/gru_cell_4/split_1SplitV1forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0-forward_gru_1/while/gru_cell_4/Const:output:09forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split½
"forward_gru_1/while/gru_cell_4/addAddV2-forward_gru_1/while/gru_cell_4/split:output:0/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru_1/while/gru_cell_4/SigmoidSigmoid&forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
$forward_gru_1/while/gru_cell_4/add_1AddV2-forward_gru_1/while/gru_cell_4/split:output:1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid(forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
"forward_gru_1/while/gru_cell_4/mulMul,forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
$forward_gru_1/while/gru_cell_4/add_2AddV2-forward_gru_1/while/gru_cell_4/split:output:2&forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#forward_gru_1/while/gru_cell_4/ReluRelu(forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
$forward_gru_1/while/gru_cell_4/mul_1Mul*forward_gru_1/while/gru_cell_4/Sigmoid:y:0!forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
$forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"forward_gru_1/while/gru_cell_4/subSub-forward_gru_1/while/gru_cell_4/sub/x:output:0*forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
$forward_gru_1/while/gru_cell_4/mul_2Mul&forward_gru_1/while/gru_cell_4/sub:z:01forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
$forward_gru_1/while/gru_cell_4/add_3AddV2(forward_gru_1/while/gru_cell_4/mul_1:z:0(forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
8forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!forward_gru_1_while_placeholder_1forward_gru_1_while_placeholder(forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/addAddV2forward_gru_1_while_placeholder"forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ]
forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/add_1AddV24forward_gru_1_while_forward_gru_1_while_loop_counter$forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_gru_1/while/IdentityIdentityforward_gru_1/while/add_1:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: ¢
forward_gru_1/while/Identity_1Identity:forward_gru_1_while_forward_gru_1_while_maximum_iterations^forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
forward_gru_1/while/Identity_2Identityforward_gru_1/while/add:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ã
forward_gru_1/while/Identity_3IdentityHforward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¡
forward_gru_1/while/Identity_4Identity(forward_gru_1/while/gru_cell_4/add_3:z:0^forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
forward_gru_1/while/NoOpNoOp5^forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp7^forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp.^forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1forward_gru_1_while_forward_gru_1_strided_slice_13forward_gru_1_while_forward_gru_1_strided_slice_1_0"
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"r
6forward_gru_1_while_gru_cell_4_readvariableop_resource8forward_gru_1_while_gru_cell_4_readvariableop_resource_0"E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0"I
forward_gru_1_while_identity_1'forward_gru_1/while/Identity_1:output:0"I
forward_gru_1_while_identity_2'forward_gru_1/while/Identity_2:output:0"I
forward_gru_1_while_identity_3'forward_gru_1/while/Identity_3:output:0"I
forward_gru_1_while_identity_4'forward_gru_1/while/Identity_4:output:0"à
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensoroforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2l
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2p
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2^
-forward_gru_1/while/gru_cell_4/ReadVariableOp-forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_45611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_45611___redundant_placeholder03
/while_while_cond_45611___redundant_placeholder13
/while_while_cond_45611___redundant_placeholder23
/while_while_cond_45611___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
b

.bidirectional_1_forward_gru_1_while_body_43320X
Tbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_loop_counter^
Zbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_maximum_iterations3
/bidirectional_1_forward_gru_1_while_placeholder5
1bidirectional_1_forward_gru_1_while_placeholder_15
1bidirectional_1_forward_gru_1_while_placeholder_2W
Sbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1_0
bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0Z
Hbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0:a
Obidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:c
Qbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
0
,bidirectional_1_forward_gru_1_while_identity2
.bidirectional_1_forward_gru_1_while_identity_12
.bidirectional_1_forward_gru_1_while_identity_22
.bidirectional_1_forward_gru_1_while_identity_32
.bidirectional_1_forward_gru_1_while_identity_4U
Qbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1
bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensorX
Fbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource:_
Mbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:a
Obidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢Dbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢Fbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp¦
Ubidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ½
Gbidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0/bidirectional_1_forward_gru_1_while_placeholder^bidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Æ
=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOpHbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0½
6bidirectional_1/forward_gru_1/while/gru_cell_4/unstackUnpackEbidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÔ
Dbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOpObidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
5bidirectional_1/forward_gru_1/while/gru_cell_4/MatMulMatMulNbidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
6bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAddBiasAdd?bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul:product:0?bidirectional_1/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>bidirectional_1/forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ°
4bidirectional_1/forward_gru_1/while/gru_cell_4/splitSplitGbidirectional_1/forward_gru_1/while/gru_cell_4/split/split_dim:output:0?bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitØ
Fbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpQbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0ö
7bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1MatMul1bidirectional_1_forward_gru_1_while_placeholder_2Nbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
8bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAddAbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1:product:0?bidirectional_1/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4bidirectional_1/forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
@bidirectional_1/forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
6bidirectional_1/forward_gru_1/while/gru_cell_4/split_1SplitVAbidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0=bidirectional_1/forward_gru_1/while/gru_cell_4/Const:output:0Ibidirectional_1/forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splití
2bidirectional_1/forward_gru_1/while/gru_cell_4/addAddV2=bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:0?bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
6bidirectional_1/forward_gru_1/while/gru_cell_4/SigmoidSigmoid6bidirectional_1/forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ï
4bidirectional_1/forward_gru_1/while/gru_cell_4/add_1AddV2=bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:1?bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
8bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid8bidirectional_1/forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ê
2bidirectional_1/forward_gru_1/while/gru_cell_4/mulMul<bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0?bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
4bidirectional_1/forward_gru_1/while/gru_cell_4/add_2AddV2=bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:26bidirectional_1/forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
3bidirectional_1/forward_gru_1/while/gru_cell_4/ReluRelu8bidirectional_1/forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ü
4bidirectional_1/forward_gru_1/while/gru_cell_4/mul_1Mul:bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid:y:01bidirectional_1_forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
4bidirectional_1/forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?æ
2bidirectional_1/forward_gru_1/while/gru_cell_4/subSub=bidirectional_1/forward_gru_1/while/gru_cell_4/sub/x:output:0:bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
è
4bidirectional_1/forward_gru_1/while/gru_cell_4/mul_2Mul6bidirectional_1/forward_gru_1/while/gru_cell_4/sub:z:0Abidirectional_1/forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ã
4bidirectional_1/forward_gru_1/while/gru_cell_4/add_3AddV28bidirectional_1/forward_gru_1/while/gru_cell_4/mul_1:z:08bidirectional_1/forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
Hbidirectional_1/forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem1bidirectional_1_forward_gru_1_while_placeholder_1/bidirectional_1_forward_gru_1_while_placeholder8bidirectional_1/forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒk
)bidirectional_1/forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¶
'bidirectional_1/forward_gru_1/while/addAddV2/bidirectional_1_forward_gru_1_while_placeholder2bidirectional_1/forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: m
+bidirectional_1/forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ß
)bidirectional_1/forward_gru_1/while/add_1AddV2Tbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_loop_counter4bidirectional_1/forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: ³
,bidirectional_1/forward_gru_1/while/IdentityIdentity-bidirectional_1/forward_gru_1/while/add_1:z:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: â
.bidirectional_1/forward_gru_1/while/Identity_1IdentityZbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_maximum_iterations)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: ³
.bidirectional_1/forward_gru_1/while/Identity_2Identity+bidirectional_1/forward_gru_1/while/add:z:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: ó
.bidirectional_1/forward_gru_1/while/Identity_3IdentityXbidirectional_1/forward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒÑ
.bidirectional_1/forward_gru_1/while/Identity_4Identity8bidirectional_1/forward_gru_1/while/gru_cell_4/add_3:z:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
(bidirectional_1/forward_gru_1/while/NoOpNoOpE^bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpG^bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp>^bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "¨
Qbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1Sbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1_0"¤
Obidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceQbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0" 
Mbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resourceObidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"
Fbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resourceHbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0"e
,bidirectional_1_forward_gru_1_while_identity5bidirectional_1/forward_gru_1/while/Identity:output:0"i
.bidirectional_1_forward_gru_1_while_identity_17bidirectional_1/forward_gru_1/while/Identity_1:output:0"i
.bidirectional_1_forward_gru_1_while_identity_27bidirectional_1/forward_gru_1/while/Identity_2:output:0"i
.bidirectional_1_forward_gru_1_while_identity_37bidirectional_1/forward_gru_1/while/Identity_3:output:0"i
.bidirectional_1_forward_gru_1_while_identity_47bidirectional_1/forward_gru_1/while/Identity_4:output:0"¢
bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensorbidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Dbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpDbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2
Fbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpFbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2~
=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ë

Â
backward_gru_1_while_cond_45195:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_2<
8backward_gru_1_while_less_backward_gru_1_strided_slice_1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_45195___redundant_placeholder0Q
Mbackward_gru_1_while_backward_gru_1_while_cond_45195___redundant_placeholder1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_45195___redundant_placeholder2Q
Mbackward_gru_1_while_backward_gru_1_while_cond_45195___redundant_placeholder3!
backward_gru_1_while_identity

backward_gru_1/while/LessLess backward_gru_1_while_placeholder8backward_gru_1_while_less_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: i
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
­

Ö
*__inference_gru_cell_5_layer_call_fn_46874

inputs
states_0
unknown:
	unknown_0:
	unknown_1:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
Õ
¥
while_cond_40986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_40986___redundant_placeholder03
/while_while_cond_40986___redundant_placeholder13
/while_while_cond_40986___redundant_placeholder23
/while_while_cond_40986___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
Ù


,__inference_sequential_1_layer_call_fn_43110
bidirectional_1_input
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_43062o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namebidirectional_1_input
Ð

¯
forward_gru_1_while_cond_427268
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_2:
6forward_gru_1_while_less_forward_gru_1_strided_slice_1O
Kforward_gru_1_while_forward_gru_1_while_cond_42726___redundant_placeholder0O
Kforward_gru_1_while_forward_gru_1_while_cond_42726___redundant_placeholder1O
Kforward_gru_1_while_forward_gru_1_while_cond_42726___redundant_placeholder2O
Kforward_gru_1_while_forward_gru_1_while_cond_42726___redundant_placeholder3 
forward_gru_1_while_identity

forward_gru_1/while/LessLessforward_gru_1_while_placeholder6forward_gru_1_while_less_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: g
forward_gru_1/while/IdentityIdentityforward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
ÉA
Ö
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41227

inputs"
gru_cell_5_41145:"
gru_cell_5_41147:"
gru_cell_5_41149:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_5/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÀ
"gru_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_5_41145gru_cell_5_41147gru_cell_5_41149*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41144n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_5_41145gru_cell_5_41147gru_cell_5_41149*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_41157*
condR
while_cond_41156*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    £
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_5_41147*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È
NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_5/StatefulPartitionedCall"gru_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø

J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44663
inputs_0B
0forward_gru_1_gru_cell_4_readvariableop_resource:I
7forward_gru_1_gru_cell_4_matmul_readvariableop_resource:K
9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
C
1backward_gru_1_gru_cell_5_readvariableop_resource:J
8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:L
:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:

identity¢/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢(backward_gru_1/gru_cell_5/ReadVariableOp¢backward_gru_1/while¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢'forward_gru_1/gru_cell_4/ReadVariableOp¢forward_gru_1/whileK
forward_gru_1/ShapeShapeinputs_0*
T0*
_output_shapes
:k
!forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru_1/strided_sliceStridedSliceforward_gru_1/Shape:output:0*forward_gru_1/strided_slice/stack:output:0,forward_gru_1/strided_slice/stack_1:output:0,forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru_1/zeros/packedPack$forward_gru_1/strided_slice:output:0%forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru_1/zerosFill#forward_gru_1/zeros/packed:output:0"forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru_1/transpose	Transposeinputs_0%forward_gru_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
forward_gru_1/Shape_1Shapeforward_gru_1/transpose:y:0*
T0*
_output_shapes
:m
#forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
forward_gru_1/strided_slice_1StridedSliceforward_gru_1/Shape_1:output:0,forward_gru_1/strided_slice_1/stack:output:0.forward_gru_1/strided_slice_1/stack_1:output:0.forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
forward_gru_1/TensorArrayV2TensorListReserve2forward_gru_1/TensorArrayV2/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
5forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru_1/transpose:y:0Lforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
forward_gru_1/strided_slice_2StridedSliceforward_gru_1/transpose:y:0,forward_gru_1/strided_slice_2/stack:output:0.forward_gru_1/strided_slice_2/stack_1:output:0.forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp0forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0
 forward_gru_1/gru_cell_4/unstackUnpack/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0»
forward_gru_1/gru_cell_4/MatMulMatMul&forward_gru_1/strided_slice_2:output:06forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 forward_gru_1/gru_cell_4/BiasAddBiasAdd)forward_gru_1/gru_cell_4/MatMul:product:0)forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
forward_gru_1/gru_cell_4/splitSplit1forward_gru_1/gru_cell_4/split/split_dim:output:0)forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitª
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0µ
!forward_gru_1/gru_cell_4/MatMul_1MatMulforward_gru_1/zeros:output:08forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
"forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd+forward_gru_1/gru_cell_4/MatMul_1:product:0)forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿu
*forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
 forward_gru_1/gru_cell_4/split_1SplitV+forward_gru_1/gru_cell_4/BiasAdd_1:output:0'forward_gru_1/gru_cell_4/Const:output:03forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split«
forward_gru_1/gru_cell_4/addAddV2'forward_gru_1/gru_cell_4/split:output:0)forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru_1/gru_cell_4/SigmoidSigmoid forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
forward_gru_1/gru_cell_4/add_1AddV2'forward_gru_1/gru_cell_4/split:output:1)forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid"forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
forward_gru_1/gru_cell_4/mulMul&forward_gru_1/gru_cell_4/Sigmoid_1:y:0)forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
forward_gru_1/gru_cell_4/add_2AddV2'forward_gru_1/gru_cell_4/split:output:2 forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru_1/gru_cell_4/ReluRelu"forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru_1/gru_cell_4/mul_1Mul$forward_gru_1/gru_cell_4/Sigmoid:y:0forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
forward_gru_1/gru_cell_4/subSub'forward_gru_1/gru_cell_4/sub/x:output:0$forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
forward_gru_1/gru_cell_4/mul_2Mul forward_gru_1/gru_cell_4/sub:z:0+forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
forward_gru_1/gru_cell_4/add_3AddV2"forward_gru_1/gru_cell_4/mul_1:z:0"forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
+forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   â
forward_gru_1/TensorArrayV2_1TensorListReserve4forward_gru_1/TensorArrayV2_1/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
forward_gru_1/whileWhile)forward_gru_1/while/loop_counter:output:0/forward_gru_1/while/maximum_iterations:output:0forward_gru_1/time:output:0&forward_gru_1/TensorArrayV2_1:handle:0forward_gru_1/zeros:output:0&forward_gru_1/strided_slice_1:output:0Eforward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00forward_gru_1_gru_cell_4_readvariableop_resource7forward_gru_1_gru_cell_4_matmul_readvariableop_resource9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
forward_gru_1_while_body_44409**
cond"R 
forward_gru_1_while_cond_44408*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
>forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   õ
0forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru_1/while:output:3Gforward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0v
#forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
forward_gru_1/strided_slice_3StridedSlice9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,forward_gru_1/strided_slice_3/stack:output:0.forward_gru_1/strided_slice_3/stack_1:output:0.forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masks
forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
forward_gru_1/transpose_1	Transpose9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'forward_gru_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
i
forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    L
backward_gru_1/ShapeShapeinputs_0*
T0*
_output_shapes
:l
"backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru_1/strided_sliceStridedSlicebackward_gru_1/Shape:output:0+backward_gru_1/strided_slice/stack:output:0-backward_gru_1/strided_slice/stack_1:output:0-backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
 
backward_gru_1/zeros/packedPack%backward_gru_1/strided_slice:output:0&backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru_1/zerosFill$backward_gru_1/zeros/packed:output:0#backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru_1/transpose	Transposeinputs_0&backward_gru_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿb
backward_gru_1/Shape_1Shapebackward_gru_1/transpose:y:0*
T0*
_output_shapes
:n
$backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
backward_gru_1/strided_slice_1StridedSlicebackward_gru_1/Shape_1:output:0-backward_gru_1/strided_slice_1/stack:output:0/backward_gru_1/strided_slice_1/stack_1:output:0/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
backward_gru_1/TensorArrayV2TensorListReserve3backward_gru_1/TensorArrayV2/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ³
backward_gru_1/ReverseV2	ReverseV2backward_gru_1/transpose:y:0&backward_gru_1/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Dbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
6backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!backward_gru_1/ReverseV2:output:0Mbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
backward_gru_1/strided_slice_2StridedSlicebackward_gru_1/transpose:y:0-backward_gru_1/strided_slice_2/stack:output:0/backward_gru_1/strided_slice_2/stack_1:output:0/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOp1backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0
!backward_gru_1/gru_cell_5/unstackUnpack0backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¨
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¾
 backward_gru_1/gru_cell_5/MatMulMatMul'backward_gru_1/strided_slice_2:output:07backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!backward_gru_1/gru_cell_5/BiasAddBiasAdd*backward_gru_1/gru_cell_5/MatMul:product:0*backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
)backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
backward_gru_1/gru_cell_5/splitSplit2backward_gru_1/gru_cell_5/split/split_dim:output:0*backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¬
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¸
"backward_gru_1/gru_cell_5/MatMul_1MatMulbackward_gru_1/zeros:output:09backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd,backward_gru_1/gru_cell_5/MatMul_1:product:0*backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿv
+backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ®
!backward_gru_1/gru_cell_5/split_1SplitV,backward_gru_1/gru_cell_5/BiasAdd_1:output:0(backward_gru_1/gru_cell_5/Const:output:04backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split®
backward_gru_1/gru_cell_5/addAddV2(backward_gru_1/gru_cell_5/split:output:0*backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru_1/gru_cell_5/SigmoidSigmoid!backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
backward_gru_1/gru_cell_5/add_1AddV2(backward_gru_1/gru_cell_5/split:output:1*backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid#backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
backward_gru_1/gru_cell_5/mulMul'backward_gru_1/gru_cell_5/Sigmoid_1:y:0*backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
backward_gru_1/gru_cell_5/add_2AddV2(backward_gru_1/gru_cell_5/split:output:2!backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru_1/gru_cell_5/ReluRelu#backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru_1/gru_cell_5/mul_1Mul%backward_gru_1/gru_cell_5/Sigmoid:y:0backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
backward_gru_1/gru_cell_5/subSub(backward_gru_1/gru_cell_5/sub/x:output:0%backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
backward_gru_1/gru_cell_5/mul_2Mul!backward_gru_1/gru_cell_5/sub:z:0,backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
backward_gru_1/gru_cell_5/add_3AddV2#backward_gru_1/gru_cell_5/mul_1:z:0#backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
,backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   å
backward_gru_1/TensorArrayV2_1TensorListReserve5backward_gru_1/TensorArrayV2_1/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü
backward_gru_1/whileWhile*backward_gru_1/while/loop_counter:output:00backward_gru_1/while/maximum_iterations:output:0backward_gru_1/time:output:0'backward_gru_1/TensorArrayV2_1:handle:0backward_gru_1/zeros:output:0'backward_gru_1/strided_slice_1:output:0Fbackward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01backward_gru_1_gru_cell_5_readvariableop_resource8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
backward_gru_1_while_body_44560*+
cond#R!
backward_gru_1_while_cond_44559*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
?backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ø
1backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru_1/while:output:3Hbackward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0w
$backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
backward_gru_1/strided_slice_3StridedSlice:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0-backward_gru_1/strided_slice_3/stack:output:0/backward_gru_1/strided_slice_3/stack_1:output:0/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskt
backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ì
backward_gru_1/transpose_1	Transpose:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0(backward_gru_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
j
backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :´
concatConcatV2&forward_gru_1/strided_slice_3:output:0'backward_gru_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ë
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp0^backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2^backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp)^backward_gru_1/gru_cell_5/ReadVariableOp^backward_gru_1/whileS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp/^forward_gru_1/gru_cell_4/MatMul/ReadVariableOp1^forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp(^forward_gru_1/gru_cell_4/ReadVariableOp^forward_gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2b
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2f
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2T
(backward_gru_1/gru_cell_5/ReadVariableOp(backward_gru_1/gru_cell_5/ReadVariableOp2,
backward_gru_1/whilebackward_gru_1/while2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2`
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2d
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2R
'forward_gru_1/gru_cell_4/ReadVariableOp'forward_gru_1/gru_cell_4/ReadVariableOp2*
forward_gru_1/whileforward_gru_1/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ë

Â
backward_gru_1_while_cond_42436:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_2<
8backward_gru_1_while_less_backward_gru_1_strided_slice_1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42436___redundant_placeholder0Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42436___redundant_placeholder1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42436___redundant_placeholder2Q
Mbackward_gru_1_while_backward_gru_1_while_cond_42436___redundant_placeholder3!
backward_gru_1_while_identity

backward_gru_1/while/LessLess backward_gru_1_while_placeholder8backward_gru_1_while_less_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: i
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
Ø

J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44345
inputs_0B
0forward_gru_1_gru_cell_4_readvariableop_resource:I
7forward_gru_1_gru_cell_4_matmul_readvariableop_resource:K
9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
C
1backward_gru_1_gru_cell_5_readvariableop_resource:J
8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:L
:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:

identity¢/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢(backward_gru_1/gru_cell_5/ReadVariableOp¢backward_gru_1/while¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢'forward_gru_1/gru_cell_4/ReadVariableOp¢forward_gru_1/whileK
forward_gru_1/ShapeShapeinputs_0*
T0*
_output_shapes
:k
!forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru_1/strided_sliceStridedSliceforward_gru_1/Shape:output:0*forward_gru_1/strided_slice/stack:output:0,forward_gru_1/strided_slice/stack_1:output:0,forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru_1/zeros/packedPack$forward_gru_1/strided_slice:output:0%forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru_1/zerosFill#forward_gru_1/zeros/packed:output:0"forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru_1/transpose	Transposeinputs_0%forward_gru_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
forward_gru_1/Shape_1Shapeforward_gru_1/transpose:y:0*
T0*
_output_shapes
:m
#forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
forward_gru_1/strided_slice_1StridedSliceforward_gru_1/Shape_1:output:0,forward_gru_1/strided_slice_1/stack:output:0.forward_gru_1/strided_slice_1/stack_1:output:0.forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
forward_gru_1/TensorArrayV2TensorListReserve2forward_gru_1/TensorArrayV2/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
5forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru_1/transpose:y:0Lforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
forward_gru_1/strided_slice_2StridedSliceforward_gru_1/transpose:y:0,forward_gru_1/strided_slice_2/stack:output:0.forward_gru_1/strided_slice_2/stack_1:output:0.forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp0forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0
 forward_gru_1/gru_cell_4/unstackUnpack/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0»
forward_gru_1/gru_cell_4/MatMulMatMul&forward_gru_1/strided_slice_2:output:06forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 forward_gru_1/gru_cell_4/BiasAddBiasAdd)forward_gru_1/gru_cell_4/MatMul:product:0)forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
forward_gru_1/gru_cell_4/splitSplit1forward_gru_1/gru_cell_4/split/split_dim:output:0)forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitª
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0µ
!forward_gru_1/gru_cell_4/MatMul_1MatMulforward_gru_1/zeros:output:08forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
"forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd+forward_gru_1/gru_cell_4/MatMul_1:product:0)forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿu
*forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
 forward_gru_1/gru_cell_4/split_1SplitV+forward_gru_1/gru_cell_4/BiasAdd_1:output:0'forward_gru_1/gru_cell_4/Const:output:03forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split«
forward_gru_1/gru_cell_4/addAddV2'forward_gru_1/gru_cell_4/split:output:0)forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru_1/gru_cell_4/SigmoidSigmoid forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
forward_gru_1/gru_cell_4/add_1AddV2'forward_gru_1/gru_cell_4/split:output:1)forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid"forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
forward_gru_1/gru_cell_4/mulMul&forward_gru_1/gru_cell_4/Sigmoid_1:y:0)forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
forward_gru_1/gru_cell_4/add_2AddV2'forward_gru_1/gru_cell_4/split:output:2 forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru_1/gru_cell_4/ReluRelu"forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru_1/gru_cell_4/mul_1Mul$forward_gru_1/gru_cell_4/Sigmoid:y:0forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
forward_gru_1/gru_cell_4/subSub'forward_gru_1/gru_cell_4/sub/x:output:0$forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
forward_gru_1/gru_cell_4/mul_2Mul forward_gru_1/gru_cell_4/sub:z:0+forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
forward_gru_1/gru_cell_4/add_3AddV2"forward_gru_1/gru_cell_4/mul_1:z:0"forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
+forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   â
forward_gru_1/TensorArrayV2_1TensorListReserve4forward_gru_1/TensorArrayV2_1/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
forward_gru_1/whileWhile)forward_gru_1/while/loop_counter:output:0/forward_gru_1/while/maximum_iterations:output:0forward_gru_1/time:output:0&forward_gru_1/TensorArrayV2_1:handle:0forward_gru_1/zeros:output:0&forward_gru_1/strided_slice_1:output:0Eforward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00forward_gru_1_gru_cell_4_readvariableop_resource7forward_gru_1_gru_cell_4_matmul_readvariableop_resource9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
forward_gru_1_while_body_44091**
cond"R 
forward_gru_1_while_cond_44090*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
>forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   õ
0forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru_1/while:output:3Gforward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0v
#forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
forward_gru_1/strided_slice_3StridedSlice9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,forward_gru_1/strided_slice_3/stack:output:0.forward_gru_1/strided_slice_3/stack_1:output:0.forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masks
forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          É
forward_gru_1/transpose_1	Transpose9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'forward_gru_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
i
forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    L
backward_gru_1/ShapeShapeinputs_0*
T0*
_output_shapes
:l
"backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru_1/strided_sliceStridedSlicebackward_gru_1/Shape:output:0+backward_gru_1/strided_slice/stack:output:0-backward_gru_1/strided_slice/stack_1:output:0-backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
 
backward_gru_1/zeros/packedPack%backward_gru_1/strided_slice:output:0&backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru_1/zerosFill$backward_gru_1/zeros/packed:output:0#backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru_1/transpose	Transposeinputs_0&backward_gru_1/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿb
backward_gru_1/Shape_1Shapebackward_gru_1/transpose:y:0*
T0*
_output_shapes
:n
$backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
backward_gru_1/strided_slice_1StridedSlicebackward_gru_1/Shape_1:output:0-backward_gru_1/strided_slice_1/stack:output:0/backward_gru_1/strided_slice_1/stack_1:output:0/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
backward_gru_1/TensorArrayV2TensorListReserve3backward_gru_1/TensorArrayV2/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ³
backward_gru_1/ReverseV2	ReverseV2backward_gru_1/transpose:y:0&backward_gru_1/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Dbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
6backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!backward_gru_1/ReverseV2:output:0Mbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
backward_gru_1/strided_slice_2StridedSlicebackward_gru_1/transpose:y:0-backward_gru_1/strided_slice_2/stack:output:0/backward_gru_1/strided_slice_2/stack_1:output:0/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOp1backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0
!backward_gru_1/gru_cell_5/unstackUnpack0backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¨
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¾
 backward_gru_1/gru_cell_5/MatMulMatMul'backward_gru_1/strided_slice_2:output:07backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!backward_gru_1/gru_cell_5/BiasAddBiasAdd*backward_gru_1/gru_cell_5/MatMul:product:0*backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
)backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
backward_gru_1/gru_cell_5/splitSplit2backward_gru_1/gru_cell_5/split/split_dim:output:0*backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¬
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¸
"backward_gru_1/gru_cell_5/MatMul_1MatMulbackward_gru_1/zeros:output:09backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd,backward_gru_1/gru_cell_5/MatMul_1:product:0*backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿv
+backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ®
!backward_gru_1/gru_cell_5/split_1SplitV,backward_gru_1/gru_cell_5/BiasAdd_1:output:0(backward_gru_1/gru_cell_5/Const:output:04backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split®
backward_gru_1/gru_cell_5/addAddV2(backward_gru_1/gru_cell_5/split:output:0*backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru_1/gru_cell_5/SigmoidSigmoid!backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
backward_gru_1/gru_cell_5/add_1AddV2(backward_gru_1/gru_cell_5/split:output:1*backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid#backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
backward_gru_1/gru_cell_5/mulMul'backward_gru_1/gru_cell_5/Sigmoid_1:y:0*backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
backward_gru_1/gru_cell_5/add_2AddV2(backward_gru_1/gru_cell_5/split:output:2!backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru_1/gru_cell_5/ReluRelu#backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru_1/gru_cell_5/mul_1Mul%backward_gru_1/gru_cell_5/Sigmoid:y:0backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
backward_gru_1/gru_cell_5/subSub(backward_gru_1/gru_cell_5/sub/x:output:0%backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
backward_gru_1/gru_cell_5/mul_2Mul!backward_gru_1/gru_cell_5/sub:z:0,backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
backward_gru_1/gru_cell_5/add_3AddV2#backward_gru_1/gru_cell_5/mul_1:z:0#backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
,backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   å
backward_gru_1/TensorArrayV2_1TensorListReserve5backward_gru_1/TensorArrayV2_1/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü
backward_gru_1/whileWhile*backward_gru_1/while/loop_counter:output:00backward_gru_1/while/maximum_iterations:output:0backward_gru_1/time:output:0'backward_gru_1/TensorArrayV2_1:handle:0backward_gru_1/zeros:output:0'backward_gru_1/strided_slice_1:output:0Fbackward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01backward_gru_1_gru_cell_5_readvariableop_resource8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
backward_gru_1_while_body_44242*+
cond#R!
backward_gru_1_while_cond_44241*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
?backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ø
1backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru_1/while:output:3Hbackward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0w
$backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
backward_gru_1/strided_slice_3StridedSlice:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0-backward_gru_1/strided_slice_3/stack:output:0/backward_gru_1/strided_slice_3/stack_1:output:0/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskt
backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ì
backward_gru_1/transpose_1	Transpose:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0(backward_gru_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
j
backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :´
concatConcatV2&forward_gru_1/strided_slice_3:output:0'backward_gru_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ë
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp0^backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2^backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp)^backward_gru_1/gru_cell_5/ReadVariableOp^backward_gru_1/whileS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp/^forward_gru_1/gru_cell_4/MatMul/ReadVariableOp1^forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp(^forward_gru_1/gru_cell_4/ReadVariableOp^forward_gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2b
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2f
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2T
(backward_gru_1/gru_cell_5/ReadVariableOp(backward_gru_1/gru_cell_5/ReadVariableOp2,
backward_gru_1/whilebackward_gru_1/while2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2`
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2d
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2R
'forward_gru_1/gru_cell_4/ReadVariableOp'forward_gru_1/gru_cell_4/ReadVariableOp2*
forward_gru_1/whileforward_gru_1/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¹<
ø
while_body_45453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_4_readvariableop_resource_0:C
1while_gru_cell_4_matmul_readvariableop_resource_0:E
3while_gru_cell_4_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_4_readvariableop_resource:A
/while_gru_cell_4_matmul_readvariableop_resource:C
1while_gru_cell_4_matmul_1_readvariableop_resource:
¢&while/gru_cell_4/MatMul/ReadVariableOp¢(while/gru_cell_4/MatMul_1/ReadVariableOp¢while/gru_cell_4/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 

ò
/bidirectional_1_backward_gru_1_while_cond_43470Z
Vbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_loop_counter`
\bidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_maximum_iterations4
0bidirectional_1_backward_gru_1_while_placeholder6
2bidirectional_1_backward_gru_1_while_placeholder_16
2bidirectional_1_backward_gru_1_while_placeholder_2\
Xbidirectional_1_backward_gru_1_while_less_bidirectional_1_backward_gru_1_strided_slice_1q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43470___redundant_placeholder0q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43470___redundant_placeholder1q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43470___redundant_placeholder2q
mbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_cond_43470___redundant_placeholder31
-bidirectional_1_backward_gru_1_while_identity
Þ
)bidirectional_1/backward_gru_1/while/LessLess0bidirectional_1_backward_gru_1_while_placeholderXbidirectional_1_backward_gru_1_while_less_bidirectional_1_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: 
-bidirectional_1/backward_gru_1/while/IdentityIdentity-bidirectional_1/backward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "g
-bidirectional_1_backward_gru_1_while_identity6bidirectional_1/backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
Û?
Ô
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_40863

inputs"
gru_cell_4_40781:"
gru_cell_4_40783:"
gru_cell_4_40785:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_4/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÀ
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_40781gru_cell_4_40783gru_cell_4_40785*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40780n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_40781gru_cell_4_40783gru_cell_4_40785*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_40793*
condR
while_cond_40792*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¢
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_4_40783*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ç
NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©


#__inference_signature_wrapper_43947
bidirectional_1_input
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallbidirectional_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_40704o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namebidirectional_1_input
³c

/bidirectional_1_backward_gru_1_while_body_43471Z
Vbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_loop_counter`
\bidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_maximum_iterations4
0bidirectional_1_backward_gru_1_while_placeholder6
2bidirectional_1_backward_gru_1_while_placeholder_16
2bidirectional_1_backward_gru_1_while_placeholder_2Y
Ubidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1_0
bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0[
Ibidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0:b
Pbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:d
Rbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
1
-bidirectional_1_backward_gru_1_while_identity3
/bidirectional_1_backward_gru_1_while_identity_13
/bidirectional_1_backward_gru_1_while_identity_23
/bidirectional_1_backward_gru_1_while_identity_33
/bidirectional_1_backward_gru_1_while_identity_4W
Sbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1
bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensorY
Gbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource:`
Nbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:b
Pbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢Ebidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢Gbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp§
Vbidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
Hbidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_00bidirectional_1_backward_gru_1_while_placeholder_bidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0È
>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOpIbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0¿
7bidirectional_1/backward_gru_1/while/gru_cell_5/unstackUnpackFbidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÖ
Ebidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOpPbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
6bidirectional_1/backward_gru_1/while/gru_cell_5/MatMulMatMulObidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Mbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
7bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAddBiasAdd@bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul:product:0@bidirectional_1/backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?bidirectional_1/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ³
5bidirectional_1/backward_gru_1/while/gru_cell_5/splitSplitHbidirectional_1/backward_gru_1/while/gru_cell_5/split/split_dim:output:0@bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÚ
Gbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpRbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0ù
8bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1MatMul2bidirectional_1_backward_gru_1_while_placeholder_2Obidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
9bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAddBbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1:product:0@bidirectional_1/backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5bidirectional_1/backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Abidirectional_1/backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
7bidirectional_1/backward_gru_1/while/gru_cell_5/split_1SplitVBbidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0>bidirectional_1/backward_gru_1/while/gru_cell_5/Const:output:0Jbidirectional_1/backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitð
3bidirectional_1/backward_gru_1/while/gru_cell_5/addAddV2>bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:0@bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
7bidirectional_1/backward_gru_1/while/gru_cell_5/SigmoidSigmoid7bidirectional_1/backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
5bidirectional_1/backward_gru_1/while/gru_cell_5/add_1AddV2>bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:1@bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
9bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid9bidirectional_1/backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
3bidirectional_1/backward_gru_1/while/gru_cell_5/mulMul=bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid_1:y:0@bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
é
5bidirectional_1/backward_gru_1/while/gru_cell_5/add_2AddV2>bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:27bidirectional_1/backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
4bidirectional_1/backward_gru_1/while/gru_cell_5/ReluRelu9bidirectional_1/backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
5bidirectional_1/backward_gru_1/while/gru_cell_5/mul_1Mul;bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid:y:02bidirectional_1_backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
5bidirectional_1/backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?é
3bidirectional_1/backward_gru_1/while/gru_cell_5/subSub>bidirectional_1/backward_gru_1/while/gru_cell_5/sub/x:output:0;bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë
5bidirectional_1/backward_gru_1/while/gru_cell_5/mul_2Mul7bidirectional_1/backward_gru_1/while/gru_cell_5/sub:z:0Bbidirectional_1/backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
5bidirectional_1/backward_gru_1/while/gru_cell_5/add_3AddV29bidirectional_1/backward_gru_1/while/gru_cell_5/mul_1:z:09bidirectional_1/backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
Ibidirectional_1/backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2bidirectional_1_backward_gru_1_while_placeholder_10bidirectional_1_backward_gru_1_while_placeholder9bidirectional_1/backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒl
*bidirectional_1/backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(bidirectional_1/backward_gru_1/while/addAddV20bidirectional_1_backward_gru_1_while_placeholder3bidirectional_1/backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: n
,bidirectional_1/backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*bidirectional_1/backward_gru_1/while/add_1AddV2Vbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_loop_counter5bidirectional_1/backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: ¶
-bidirectional_1/backward_gru_1/while/IdentityIdentity.bidirectional_1/backward_gru_1/while/add_1:z:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: æ
/bidirectional_1/backward_gru_1/while/Identity_1Identity\bidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_maximum_iterations*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¶
/bidirectional_1/backward_gru_1/while/Identity_2Identity,bidirectional_1/backward_gru_1/while/add:z:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: ö
/bidirectional_1/backward_gru_1/while/Identity_3IdentityYbidirectional_1/backward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒÔ
/bidirectional_1/backward_gru_1/while/Identity_4Identity9bidirectional_1/backward_gru_1/while/gru_cell_5/add_3:z:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¾
)bidirectional_1/backward_gru_1/while/NoOpNoOpF^bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpH^bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp?^bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "¬
Sbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1Ubidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1_0"¦
Pbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceRbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"¢
Nbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resourcePbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"
Gbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resourceIbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0"g
-bidirectional_1_backward_gru_1_while_identity6bidirectional_1/backward_gru_1/while/Identity:output:0"k
/bidirectional_1_backward_gru_1_while_identity_18bidirectional_1/backward_gru_1/while/Identity_1:output:0"k
/bidirectional_1_backward_gru_1_while_identity_28bidirectional_1/backward_gru_1/while/Identity_2:output:0"k
/bidirectional_1_backward_gru_1_while_identity_38bidirectional_1/backward_gru_1/while/Identity_3:output:0"k
/bidirectional_1_backward_gru_1_while_identity_48bidirectional_1/backward_gru_1/while/Identity_4:output:0"¦
bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensorbidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Ebidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpEbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2
Gbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpGbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2
>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Â<
ø
while_body_45771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_4_readvariableop_resource_0:C
1while_gru_cell_4_matmul_readvariableop_resource_0:E
3while_gru_cell_4_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_4_readvariableop_resource:A
/while_gru_cell_4_matmul_readvariableop_resource:C
1while_gru_cell_4_matmul_1_readvariableop_resource:
¢&while/gru_cell_4/MatMul/ReadVariableOp¢(while/gru_cell_4/MatMul_1/ReadVariableOp¢while/gru_cell_4/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Ð

¯
forward_gru_1_while_cond_450448
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_2:
6forward_gru_1_while_less_forward_gru_1_strided_slice_1O
Kforward_gru_1_while_forward_gru_1_while_cond_45044___redundant_placeholder0O
Kforward_gru_1_while_forward_gru_1_while_cond_45044___redundant_placeholder1O
Kforward_gru_1_while_forward_gru_1_while_cond_45044___redundant_placeholder2O
Kforward_gru_1_while_forward_gru_1_while_cond_45044___redundant_placeholder3 
forward_gru_1_while_identity

forward_gru_1/while/LessLessforward_gru_1_while_placeholder6forward_gru_1_while_less_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: g
forward_gru_1/while/IdentityIdentityforward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
[
Û
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41977

inputs4
"gru_cell_5_readvariableop_resource:;
)gru_cell_5_matmul_readvariableop_resource:=
+gru_cell_5_matmul_1_readvariableop_resource:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_5/MatMul/ReadVariableOp¢"gru_cell_5/MatMul_1/ReadVariableOp¢gru_cell_5/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_41882*
condR
while_cond_41881*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¼
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

Â
backward_gru_1_while_cond_44877:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_2<
8backward_gru_1_while_less_backward_gru_1_strided_slice_1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44877___redundant_placeholder0Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44877___redundant_placeholder1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44877___redundant_placeholder2Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44877___redundant_placeholder3!
backward_gru_1_while_identity

backward_gru_1/while/LessLess backward_gru_1_while_placeholder8backward_gru_1_while_less_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: i
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
¾
Ú
G__inference_sequential_1_layer_call_and_return_conditional_losses_43920

inputsR
@bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resource:Y
Gbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource:[
Ibidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
S
Abidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resource:Z
Hbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource:\
Jbidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity¢?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢Abidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢$bidirectional_1/backward_gru_1/while¢>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢#bidirectional_1/forward_gru_1/while¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOpY
#bidirectional_1/forward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:{
1bidirectional_1/forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional_1/forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3bidirectional_1/forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+bidirectional_1/forward_gru_1/strided_sliceStridedSlice,bidirectional_1/forward_gru_1/Shape:output:0:bidirectional_1/forward_gru_1/strided_slice/stack:output:0<bidirectional_1/forward_gru_1/strided_slice/stack_1:output:0<bidirectional_1/forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,bidirectional_1/forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Í
*bidirectional_1/forward_gru_1/zeros/packedPack4bidirectional_1/forward_gru_1/strided_slice:output:05bidirectional_1/forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:n
)bidirectional_1/forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Æ
#bidirectional_1/forward_gru_1/zerosFill3bidirectional_1/forward_gru_1/zeros/packed:output:02bidirectional_1/forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

,bidirectional_1/forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
'bidirectional_1/forward_gru_1/transpose	Transposeinputs5bidirectional_1/forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%bidirectional_1/forward_gru_1/Shape_1Shape+bidirectional_1/forward_gru_1/transpose:y:0*
T0*
_output_shapes
:}
3bidirectional_1/forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5bidirectional_1/forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_1/forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-bidirectional_1/forward_gru_1/strided_slice_1StridedSlice.bidirectional_1/forward_gru_1/Shape_1:output:0<bidirectional_1/forward_gru_1/strided_slice_1/stack:output:0>bidirectional_1/forward_gru_1/strided_slice_1/stack_1:output:0>bidirectional_1/forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
9bidirectional_1/forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+bidirectional_1/forward_gru_1/TensorArrayV2TensorListReserveBbidirectional_1/forward_gru_1/TensorArrayV2/element_shape:output:06bidirectional_1/forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¤
Sbidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   º
Ebidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor+bidirectional_1/forward_gru_1/transpose:y:0\bidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ}
3bidirectional_1/forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5bidirectional_1/forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_1/forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ÿ
-bidirectional_1/forward_gru_1/strided_slice_2StridedSlice+bidirectional_1/forward_gru_1/transpose:y:0<bidirectional_1/forward_gru_1/strided_slice_2/stack:output:0>bidirectional_1/forward_gru_1/strided_slice_2/stack_1:output:0>bidirectional_1/forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¸
7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp@bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0±
0bidirectional_1/forward_gru_1/gru_cell_4/unstackUnpack?bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÆ
>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOpGbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ë
/bidirectional_1/forward_gru_1/gru_cell_4/MatMulMatMul6bidirectional_1/forward_gru_1/strided_slice_2:output:0Fbidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
0bidirectional_1/forward_gru_1/gru_cell_4/BiasAddBiasAdd9bidirectional_1/forward_gru_1/gru_cell_4/MatMul:product:09bidirectional_1/forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8bidirectional_1/forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
.bidirectional_1/forward_gru_1/gru_cell_4/splitSplitAbidirectional_1/forward_gru_1/gru_cell_4/split/split_dim:output:09bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÊ
@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpIbidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0å
1bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1MatMul,bidirectional_1/forward_gru_1/zeros:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
2bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd;bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1:product:09bidirectional_1/forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.bidirectional_1/forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
:bidirectional_1/forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
0bidirectional_1/forward_gru_1/gru_cell_4/split_1SplitV;bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd_1:output:07bidirectional_1/forward_gru_1/gru_cell_4/Const:output:0Cbidirectional_1/forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÛ
,bidirectional_1/forward_gru_1/gru_cell_4/addAddV27bidirectional_1/forward_gru_1/gru_cell_4/split:output:09bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0bidirectional_1/forward_gru_1/gru_cell_4/SigmoidSigmoid0bidirectional_1/forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
.bidirectional_1/forward_gru_1/gru_cell_4/add_1AddV27bidirectional_1/forward_gru_1/gru_cell_4/split:output:19bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid2bidirectional_1/forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ø
,bidirectional_1/forward_gru_1/gru_cell_4/mulMul6bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid_1:y:09bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô
.bidirectional_1/forward_gru_1/gru_cell_4/add_2AddV27bidirectional_1/forward_gru_1/gru_cell_4/split:output:20bidirectional_1/forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-bidirectional_1/forward_gru_1/gru_cell_4/ReluRelu2bidirectional_1/forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ë
.bidirectional_1/forward_gru_1/gru_cell_4/mul_1Mul4bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid:y:0,bidirectional_1/forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
.bidirectional_1/forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ô
,bidirectional_1/forward_gru_1/gru_cell_4/subSub7bidirectional_1/forward_gru_1/gru_cell_4/sub/x:output:04bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
.bidirectional_1/forward_gru_1/gru_cell_4/mul_2Mul0bidirectional_1/forward_gru_1/gru_cell_4/sub:z:0;bidirectional_1/forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ñ
.bidirectional_1/forward_gru_1/gru_cell_4/add_3AddV22bidirectional_1/forward_gru_1/gru_cell_4/mul_1:z:02bidirectional_1/forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;bidirectional_1/forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
-bidirectional_1/forward_gru_1/TensorArrayV2_1TensorListReserveDbidirectional_1/forward_gru_1/TensorArrayV2_1/element_shape:output:06bidirectional_1/forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
"bidirectional_1/forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
6bidirectional_1/forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿr
0bidirectional_1/forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
#bidirectional_1/forward_gru_1/whileWhile9bidirectional_1/forward_gru_1/while/loop_counter:output:0?bidirectional_1/forward_gru_1/while/maximum_iterations:output:0+bidirectional_1/forward_gru_1/time:output:06bidirectional_1/forward_gru_1/TensorArrayV2_1:handle:0,bidirectional_1/forward_gru_1/zeros:output:06bidirectional_1/forward_gru_1/strided_slice_1:output:0Ubidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resourceGbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resourceIbidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.bidirectional_1_forward_gru_1_while_body_43652*:
cond2R0
.bidirectional_1_forward_gru_1_while_cond_43651*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
Nbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
@bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStack,bidirectional_1/forward_gru_1/while:output:3Wbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
3bidirectional_1/forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
5bidirectional_1/forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5bidirectional_1/forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
-bidirectional_1/forward_gru_1/strided_slice_3StridedSliceIbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0<bidirectional_1/forward_gru_1/strided_slice_3/stack:output:0>bidirectional_1/forward_gru_1/strided_slice_3/stack_1:output:0>bidirectional_1/forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
.bidirectional_1/forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ð
)bidirectional_1/forward_gru_1/transpose_1	TransposeIbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:07bidirectional_1/forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
%bidirectional_1/forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Z
$bidirectional_1/backward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:|
2bidirectional_1/backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4bidirectional_1/backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4bidirectional_1/backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,bidirectional_1/backward_gru_1/strided_sliceStridedSlice-bidirectional_1/backward_gru_1/Shape:output:0;bidirectional_1/backward_gru_1/strided_slice/stack:output:0=bidirectional_1/backward_gru_1/strided_slice/stack_1:output:0=bidirectional_1/backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-bidirectional_1/backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Ð
+bidirectional_1/backward_gru_1/zeros/packedPack5bidirectional_1/backward_gru_1/strided_slice:output:06bidirectional_1/backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:o
*bidirectional_1/backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    É
$bidirectional_1/backward_gru_1/zerosFill4bidirectional_1/backward_gru_1/zeros/packed:output:03bidirectional_1/backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-bidirectional_1/backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
(bidirectional_1/backward_gru_1/transpose	Transposeinputs6bidirectional_1/backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&bidirectional_1/backward_gru_1/Shape_1Shape,bidirectional_1/backward_gru_1/transpose:y:0*
T0*
_output_shapes
:~
4bidirectional_1/backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6bidirectional_1/backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_1/backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.bidirectional_1/backward_gru_1/strided_slice_1StridedSlice/bidirectional_1/backward_gru_1/Shape_1:output:0=bidirectional_1/backward_gru_1/strided_slice_1/stack:output:0?bidirectional_1/backward_gru_1/strided_slice_1/stack_1:output:0?bidirectional_1/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:bidirectional_1/backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,bidirectional_1/backward_gru_1/TensorArrayV2TensorListReserveCbidirectional_1/backward_gru_1/TensorArrayV2/element_shape:output:07bidirectional_1/backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒw
-bidirectional_1/backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ñ
(bidirectional_1/backward_gru_1/ReverseV2	ReverseV2,bidirectional_1/backward_gru_1/transpose:y:06bidirectional_1/backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Tbidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
Fbidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor1bidirectional_1/backward_gru_1/ReverseV2:output:0]bidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ~
4bidirectional_1/backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6bidirectional_1/backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_1/backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
.bidirectional_1/backward_gru_1/strided_slice_2StridedSlice,bidirectional_1/backward_gru_1/transpose:y:0=bidirectional_1/backward_gru_1/strided_slice_2/stack:output:0?bidirectional_1/backward_gru_1/strided_slice_2/stack_1:output:0?bidirectional_1/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskº
8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOpAbidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0³
1bidirectional_1/backward_gru_1/gru_cell_5/unstackUnpack@bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÈ
?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOpHbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0î
0bidirectional_1/backward_gru_1/gru_cell_5/MatMulMatMul7bidirectional_1/backward_gru_1/strided_slice_2:output:0Gbidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
1bidirectional_1/backward_gru_1/gru_cell_5/BiasAddBiasAdd:bidirectional_1/backward_gru_1/gru_cell_5/MatMul:product:0:bidirectional_1/backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9bidirectional_1/backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¡
/bidirectional_1/backward_gru_1/gru_cell_5/splitSplitBbidirectional_1/backward_gru_1/gru_cell_5/split/split_dim:output:0:bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÌ
Abidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpJbidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0è
2bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1MatMul-bidirectional_1/backward_gru_1/zeros:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
3bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd<bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1:product:0:bidirectional_1/backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/bidirectional_1/backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
;bidirectional_1/backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
1bidirectional_1/backward_gru_1/gru_cell_5/split_1SplitV<bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd_1:output:08bidirectional_1/backward_gru_1/gru_cell_5/Const:output:0Dbidirectional_1/backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÞ
-bidirectional_1/backward_gru_1/gru_cell_5/addAddV28bidirectional_1/backward_gru_1/gru_cell_5/split:output:0:bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
1bidirectional_1/backward_gru_1/gru_cell_5/SigmoidSigmoid1bidirectional_1/backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à
/bidirectional_1/backward_gru_1/gru_cell_5/add_1AddV28bidirectional_1/backward_gru_1/gru_cell_5/split:output:1:bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
3bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid3bidirectional_1/backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Û
-bidirectional_1/backward_gru_1/gru_cell_5/mulMul7bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid_1:y:0:bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
/bidirectional_1/backward_gru_1/gru_cell_5/add_2AddV28bidirectional_1/backward_gru_1/gru_cell_5/split:output:21bidirectional_1/backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

.bidirectional_1/backward_gru_1/gru_cell_5/ReluRelu3bidirectional_1/backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Î
/bidirectional_1/backward_gru_1/gru_cell_5/mul_1Mul5bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid:y:0-bidirectional_1/backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t
/bidirectional_1/backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?×
-bidirectional_1/backward_gru_1/gru_cell_5/subSub8bidirectional_1/backward_gru_1/gru_cell_5/sub/x:output:05bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ù
/bidirectional_1/backward_gru_1/gru_cell_5/mul_2Mul1bidirectional_1/backward_gru_1/gru_cell_5/sub:z:0<bidirectional_1/backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô
/bidirectional_1/backward_gru_1/gru_cell_5/add_3AddV23bidirectional_1/backward_gru_1/gru_cell_5/mul_1:z:03bidirectional_1/backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<bidirectional_1/backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
.bidirectional_1/backward_gru_1/TensorArrayV2_1TensorListReserveEbidirectional_1/backward_gru_1/TensorArrayV2_1/element_shape:output:07bidirectional_1/backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
#bidirectional_1/backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
7bidirectional_1/backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1bidirectional_1/backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ì
$bidirectional_1/backward_gru_1/whileWhile:bidirectional_1/backward_gru_1/while/loop_counter:output:0@bidirectional_1/backward_gru_1/while/maximum_iterations:output:0,bidirectional_1/backward_gru_1/time:output:07bidirectional_1/backward_gru_1/TensorArrayV2_1:handle:0-bidirectional_1/backward_gru_1/zeros:output:07bidirectional_1/backward_gru_1/strided_slice_1:output:0Vbidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Abidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resourceHbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resourceJbidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *;
body3R1
/bidirectional_1_backward_gru_1_while_body_43803*;
cond3R1
/bidirectional_1_backward_gru_1_while_cond_43802*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations  
Obidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
Abidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStack-bidirectional_1/backward_gru_1/while:output:3Xbidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
4bidirectional_1/backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
6bidirectional_1/backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6bidirectional_1/backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
.bidirectional_1/backward_gru_1/strided_slice_3StridedSliceJbidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0=bidirectional_1/backward_gru_1/strided_slice_3/stack:output:0?bidirectional_1/backward_gru_1/strided_slice_3/stack_1:output:0?bidirectional_1/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
/bidirectional_1/backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ó
*bidirectional_1/backward_gru_1/transpose_1	TransposeJbidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:08bidirectional_1/backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
&bidirectional_1/backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ô
bidirectional_1/concatConcatV26bidirectional_1/forward_gru_1/strided_slice_3:output:07bidirectional_1/backward_gru_1/strided_slice_3:output:0$bidirectional_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMulbidirectional_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpGbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Û
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp@^bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpB^bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp9^bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp%^bidirectional_1/backward_gru_1/while?^bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpA^bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp8^bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp$^bidirectional_1/forward_gru_1/while^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2
?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2
Abidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpAbidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2t
8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2L
$bidirectional_1/backward_gru_1/while$bidirectional_1/backward_gru_1/while2
>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2
@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2r
7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2J
#bidirectional_1/forward_gru_1/while#bidirectional_1/forward_gru_1/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù


,__inference_sequential_1_layer_call_fn_42624
bidirectional_1_input
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_42601o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namebidirectional_1_input
Ð

¯
forward_gru_1_while_cond_440908
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_2:
6forward_gru_1_while_less_forward_gru_1_strided_slice_1O
Kforward_gru_1_while_forward_gru_1_while_cond_44090___redundant_placeholder0O
Kforward_gru_1_while_forward_gru_1_while_cond_44090___redundant_placeholder1O
Kforward_gru_1_while_forward_gru_1_while_cond_44090___redundant_placeholder2O
Kforward_gru_1_while_forward_gru_1_while_cond_44090___redundant_placeholder3 
forward_gru_1_while_identity

forward_gru_1/while/LessLessforward_gru_1_while_placeholder6forward_gru_1_while_less_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: g
forward_gru_1/while/IdentityIdentityforward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:

ß
.bidirectional_1_forward_gru_1_while_cond_43319X
Tbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_loop_counter^
Zbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_maximum_iterations3
/bidirectional_1_forward_gru_1_while_placeholder5
1bidirectional_1_forward_gru_1_while_placeholder_15
1bidirectional_1_forward_gru_1_while_placeholder_2Z
Vbidirectional_1_forward_gru_1_while_less_bidirectional_1_forward_gru_1_strided_slice_1o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43319___redundant_placeholder0o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43319___redundant_placeholder1o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43319___redundant_placeholder2o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43319___redundant_placeholder30
,bidirectional_1_forward_gru_1_while_identity
Ú
(bidirectional_1/forward_gru_1/while/LessLess/bidirectional_1_forward_gru_1_while_placeholderVbidirectional_1_forward_gru_1_while_less_bidirectional_1_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: 
,bidirectional_1/forward_gru_1/while/IdentityIdentity,bidirectional_1/forward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "e
,bidirectional_1_forward_gru_1_while_identity5bidirectional_1/forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:

ß
.bidirectional_1_forward_gru_1_while_cond_43651X
Tbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_loop_counter^
Zbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_maximum_iterations3
/bidirectional_1_forward_gru_1_while_placeholder5
1bidirectional_1_forward_gru_1_while_placeholder_15
1bidirectional_1_forward_gru_1_while_placeholder_2Z
Vbidirectional_1_forward_gru_1_while_less_bidirectional_1_forward_gru_1_strided_slice_1o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43651___redundant_placeholder0o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43651___redundant_placeholder1o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43651___redundant_placeholder2o
kbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_cond_43651___redundant_placeholder30
,bidirectional_1_forward_gru_1_while_identity
Ú
(bidirectional_1/forward_gru_1/while/LessLess/bidirectional_1_forward_gru_1_while_placeholderVbidirectional_1_forward_gru_1_while_less_bidirectional_1_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: 
,bidirectional_1/forward_gru_1/while/IdentityIdentity,bidirectional_1/forward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "e
,bidirectional_1_forward_gru_1_while_identity5bidirectional_1/forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:


ó
B__inference_dense_3_layer_call_and_return_conditional_losses_45339

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_46301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46301___redundant_placeholder03
/while_while_cond_46301___redundant_placeholder13
/while_while_cond_46301___redundant_placeholder23
/while_while_cond_46301___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:

½
-__inference_forward_gru_1_layer_call_fn_45367
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_41057o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¹<
ø
while_body_45612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_4_readvariableop_resource_0:C
1while_gru_cell_4_matmul_readvariableop_resource_0:E
3while_gru_cell_4_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_4_readvariableop_resource:A
/while_gru_cell_4_matmul_readvariableop_resource:C
1while_gru_cell_4_matmul_1_readvariableop_resource:
¢&while/gru_cell_4/MatMul/ReadVariableOp¢(while/gru_cell_4/MatMul_1/ReadVariableOp¢while/gru_cell_4/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ëÖ

J__inference_bidirectional_1_layer_call_and_return_conditional_losses_45299

inputsB
0forward_gru_1_gru_cell_4_readvariableop_resource:I
7forward_gru_1_gru_cell_4_matmul_readvariableop_resource:K
9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
C
1backward_gru_1_gru_cell_5_readvariableop_resource:J
8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:L
:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:

identity¢/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢(backward_gru_1/gru_cell_5/ReadVariableOp¢backward_gru_1/while¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢'forward_gru_1/gru_cell_4/ReadVariableOp¢forward_gru_1/whileI
forward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:k
!forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru_1/strided_sliceStridedSliceforward_gru_1/Shape:output:0*forward_gru_1/strided_slice/stack:output:0,forward_gru_1/strided_slice/stack_1:output:0,forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru_1/zeros/packedPack$forward_gru_1/strided_slice:output:0%forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru_1/zerosFill#forward_gru_1/zeros/packed:output:0"forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru_1/transpose	Transposeinputs%forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
forward_gru_1/Shape_1Shapeforward_gru_1/transpose:y:0*
T0*
_output_shapes
:m
#forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
forward_gru_1/strided_slice_1StridedSliceforward_gru_1/Shape_1:output:0,forward_gru_1/strided_slice_1/stack:output:0.forward_gru_1/strided_slice_1/stack_1:output:0.forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
forward_gru_1/TensorArrayV2TensorListReserve2forward_gru_1/TensorArrayV2/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Cforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
5forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru_1/transpose:y:0Lforward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒm
#forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
forward_gru_1/strided_slice_2StridedSliceforward_gru_1/transpose:y:0,forward_gru_1/strided_slice_2/stack:output:0.forward_gru_1/strided_slice_2/stack_1:output:0.forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
'forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp0forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0
 forward_gru_1/gru_cell_4/unstackUnpack/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0»
forward_gru_1/gru_cell_4/MatMulMatMul&forward_gru_1/strided_slice_2:output:06forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
 forward_gru_1/gru_cell_4/BiasAddBiasAdd)forward_gru_1/gru_cell_4/MatMul:product:0)forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
forward_gru_1/gru_cell_4/splitSplit1forward_gru_1/gru_cell_4/split/split_dim:output:0)forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitª
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0µ
!forward_gru_1/gru_cell_4/MatMul_1MatMulforward_gru_1/zeros:output:08forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
"forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd+forward_gru_1/gru_cell_4/MatMul_1:product:0)forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿu
*forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
 forward_gru_1/gru_cell_4/split_1SplitV+forward_gru_1/gru_cell_4/BiasAdd_1:output:0'forward_gru_1/gru_cell_4/Const:output:03forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split«
forward_gru_1/gru_cell_4/addAddV2'forward_gru_1/gru_cell_4/split:output:0)forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru_1/gru_cell_4/SigmoidSigmoid forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
forward_gru_1/gru_cell_4/add_1AddV2'forward_gru_1/gru_cell_4/split:output:1)forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid"forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
forward_gru_1/gru_cell_4/mulMul&forward_gru_1/gru_cell_4/Sigmoid_1:y:0)forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
forward_gru_1/gru_cell_4/add_2AddV2'forward_gru_1/gru_cell_4/split:output:2 forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru_1/gru_cell_4/ReluRelu"forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru_1/gru_cell_4/mul_1Mul$forward_gru_1/gru_cell_4/Sigmoid:y:0forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
forward_gru_1/gru_cell_4/subSub'forward_gru_1/gru_cell_4/sub/x:output:0$forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
forward_gru_1/gru_cell_4/mul_2Mul forward_gru_1/gru_cell_4/sub:z:0+forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
forward_gru_1/gru_cell_4/add_3AddV2"forward_gru_1/gru_cell_4/mul_1:z:0"forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
+forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   â
forward_gru_1/TensorArrayV2_1TensorListReserve4forward_gru_1/TensorArrayV2_1/element_shape:output:0&forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒT
forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿb
 forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
forward_gru_1/whileWhile)forward_gru_1/while/loop_counter:output:0/forward_gru_1/while/maximum_iterations:output:0forward_gru_1/time:output:0&forward_gru_1/TensorArrayV2_1:handle:0forward_gru_1/zeros:output:0&forward_gru_1/strided_slice_1:output:0Eforward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00forward_gru_1_gru_cell_4_readvariableop_resource7forward_gru_1_gru_cell_4_matmul_readvariableop_resource9forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
forward_gru_1_while_body_45045**
cond"R 
forward_gru_1_while_cond_45044*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
>forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ì
0forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru_1/while:output:3Gforward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0v
#forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
forward_gru_1/strided_slice_3StridedSlice9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0,forward_gru_1/strided_slice_3/stack:output:0.forward_gru_1/strided_slice_3/stack_1:output:0.forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masks
forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          À
forward_gru_1/transpose_1	Transpose9forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0'forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    J
backward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:l
"backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru_1/strided_sliceStridedSlicebackward_gru_1/Shape:output:0+backward_gru_1/strided_slice/stack:output:0-backward_gru_1/strided_slice/stack_1:output:0-backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
 
backward_gru_1/zeros/packedPack%backward_gru_1/strided_slice:output:0&backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru_1/zerosFill$backward_gru_1/zeros/packed:output:0#backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru_1/transpose	Transposeinputs&backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
backward_gru_1/Shape_1Shapebackward_gru_1/transpose:y:0*
T0*
_output_shapes
:n
$backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
backward_gru_1/strided_slice_1StridedSlicebackward_gru_1/Shape_1:output:0-backward_gru_1/strided_slice_1/stack:output:0/backward_gru_1/strided_slice_1/stack_1:output:0/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
backward_gru_1/TensorArrayV2TensorListReserve3backward_gru_1/TensorArrayV2/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¡
backward_gru_1/ReverseV2	ReverseV2backward_gru_1/transpose:y:0&backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!backward_gru_1/ReverseV2:output:0Mbackward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
backward_gru_1/strided_slice_2StridedSlicebackward_gru_1/transpose:y:0-backward_gru_1/strided_slice_2/stack:output:0/backward_gru_1/strided_slice_2/stack_1:output:0/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOp1backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0
!backward_gru_1/gru_cell_5/unstackUnpack0backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¨
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¾
 backward_gru_1/gru_cell_5/MatMulMatMul'backward_gru_1/strided_slice_2:output:07backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!backward_gru_1/gru_cell_5/BiasAddBiasAdd*backward_gru_1/gru_cell_5/MatMul:product:0*backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
)backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿñ
backward_gru_1/gru_cell_5/splitSplit2backward_gru_1/gru_cell_5/split/split_dim:output:0*backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¬
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¸
"backward_gru_1/gru_cell_5/MatMul_1MatMulbackward_gru_1/zeros:output:09backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
#backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd,backward_gru_1/gru_cell_5/MatMul_1:product:0*backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿv
+backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ®
!backward_gru_1/gru_cell_5/split_1SplitV,backward_gru_1/gru_cell_5/BiasAdd_1:output:0(backward_gru_1/gru_cell_5/Const:output:04backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split®
backward_gru_1/gru_cell_5/addAddV2(backward_gru_1/gru_cell_5/split:output:0*backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru_1/gru_cell_5/SigmoidSigmoid!backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
backward_gru_1/gru_cell_5/add_1AddV2(backward_gru_1/gru_cell_5/split:output:1*backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid#backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
backward_gru_1/gru_cell_5/mulMul'backward_gru_1/gru_cell_5/Sigmoid_1:y:0*backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
backward_gru_1/gru_cell_5/add_2AddV2(backward_gru_1/gru_cell_5/split:output:2!backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru_1/gru_cell_5/ReluRelu#backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru_1/gru_cell_5/mul_1Mul%backward_gru_1/gru_cell_5/Sigmoid:y:0backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?§
backward_gru_1/gru_cell_5/subSub(backward_gru_1/gru_cell_5/sub/x:output:0%backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
backward_gru_1/gru_cell_5/mul_2Mul!backward_gru_1/gru_cell_5/sub:z:0,backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
backward_gru_1/gru_cell_5/add_3AddV2#backward_gru_1/gru_cell_5/mul_1:z:0#backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
,backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   å
backward_gru_1/TensorArrayV2_1TensorListReserve5backward_gru_1/TensorArrayV2_1/element_shape:output:0'backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü
backward_gru_1/whileWhile*backward_gru_1/while/loop_counter:output:00backward_gru_1/while/maximum_iterations:output:0backward_gru_1/time:output:0'backward_gru_1/TensorArrayV2_1:handle:0backward_gru_1/zeros:output:0'backward_gru_1/strided_slice_1:output:0Fbackward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:01backward_gru_1_gru_cell_5_readvariableop_resource8backward_gru_1_gru_cell_5_matmul_readvariableop_resource:backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
backward_gru_1_while_body_45196*+
cond#R!
backward_gru_1_while_cond_45195*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
?backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ï
1backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru_1/while:output:3Hbackward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0w
$backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
backward_gru_1/strided_slice_3StridedSlice:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0-backward_gru_1/strided_slice_3/stack:output:0/backward_gru_1/strided_slice_3/stack_1:output:0/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskt
backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
backward_gru_1/transpose_1	Transpose:backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0(backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :´
concatConcatV2&forward_gru_1/strided_slice_3:output:0'backward_gru_1/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ë
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp0^backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2^backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp)^backward_gru_1/gru_cell_5/ReadVariableOp^backward_gru_1/whileS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp/^forward_gru_1/gru_cell_4/MatMul/ReadVariableOp1^forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp(^forward_gru_1/gru_cell_4/ReadVariableOp^forward_gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2b
/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2f
1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp1backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2T
(backward_gru_1/gru_cell_5/ReadVariableOp(backward_gru_1/gru_cell_5/ReadVariableOp2,
backward_gru_1/whilebackward_gru_1/while2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2`
.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp.forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2d
0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp0forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2R
'forward_gru_1/gru_cell_4/ReadVariableOp'forward_gru_1/gru_cell_4/ReadVariableOp2*
forward_gru_1/whileforward_gru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â<
ø
while_body_45930
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_4_readvariableop_resource_0:C
1while_gru_cell_4_matmul_readvariableop_resource_0:E
3while_gru_cell_4_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_4_readvariableop_resource:A
/while_gru_cell_4_matmul_readvariableop_resource:C
1while_gru_cell_4_matmul_1_readvariableop_resource:
¢&while/gru_cell_4/MatMul/ReadVariableOp¢(while/gru_cell_4/MatMul_1/ReadVariableOp¢while/gru_cell_4/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ª,
ï
G__inference_sequential_1_layer_call_and_return_conditional_losses_42601

inputs'
bidirectional_1_42541:'
bidirectional_1_42543:'
bidirectional_1_42545:
'
bidirectional_1_42547:'
bidirectional_1_42549:'
bidirectional_1_42551:

dense_2_42566:
dense_2_42568:
dense_3_42583:
dense_3_42585:
identity¢'bidirectional_1/StatefulPartitionedCall¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallí
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_1_42541bidirectional_1_42543bidirectional_1_42545bidirectional_1_42547bidirectional_1_42549bidirectional_1_42551*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42540
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_42566dense_2_42568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42565
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_42583dense_3_42585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42582§
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_42543*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_42549*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOp(^bidirectional_1/StatefulPartitionedCallS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³c

/bidirectional_1_backward_gru_1_while_body_43803Z
Vbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_loop_counter`
\bidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_maximum_iterations4
0bidirectional_1_backward_gru_1_while_placeholder6
2bidirectional_1_backward_gru_1_while_placeholder_16
2bidirectional_1_backward_gru_1_while_placeholder_2Y
Ubidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1_0
bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0[
Ibidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0:b
Pbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:d
Rbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
1
-bidirectional_1_backward_gru_1_while_identity3
/bidirectional_1_backward_gru_1_while_identity_13
/bidirectional_1_backward_gru_1_while_identity_23
/bidirectional_1_backward_gru_1_while_identity_33
/bidirectional_1_backward_gru_1_while_identity_4W
Sbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1
bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensorY
Gbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource:`
Nbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:b
Pbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢Ebidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢Gbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp§
Vbidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
Hbidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_00bidirectional_1_backward_gru_1_while_placeholder_bidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0È
>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOpIbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0¿
7bidirectional_1/backward_gru_1/while/gru_cell_5/unstackUnpackFbidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÖ
Ebidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOpPbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
6bidirectional_1/backward_gru_1/while/gru_cell_5/MatMulMatMulObidirectional_1/backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Mbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
7bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAddBiasAdd@bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul:product:0@bidirectional_1/backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?bidirectional_1/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ³
5bidirectional_1/backward_gru_1/while/gru_cell_5/splitSplitHbidirectional_1/backward_gru_1/while/gru_cell_5/split/split_dim:output:0@bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÚ
Gbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpRbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0ù
8bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1MatMul2bidirectional_1_backward_gru_1_while_placeholder_2Obidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
9bidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAddBbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1:product:0@bidirectional_1/backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5bidirectional_1/backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Abidirectional_1/backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
7bidirectional_1/backward_gru_1/while/gru_cell_5/split_1SplitVBbidirectional_1/backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0>bidirectional_1/backward_gru_1/while/gru_cell_5/Const:output:0Jbidirectional_1/backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitð
3bidirectional_1/backward_gru_1/while/gru_cell_5/addAddV2>bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:0@bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
7bidirectional_1/backward_gru_1/while/gru_cell_5/SigmoidSigmoid7bidirectional_1/backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
5bidirectional_1/backward_gru_1/while/gru_cell_5/add_1AddV2>bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:1@bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
9bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid9bidirectional_1/backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
3bidirectional_1/backward_gru_1/while/gru_cell_5/mulMul=bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid_1:y:0@bidirectional_1/backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
é
5bidirectional_1/backward_gru_1/while/gru_cell_5/add_2AddV2>bidirectional_1/backward_gru_1/while/gru_cell_5/split:output:27bidirectional_1/backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
4bidirectional_1/backward_gru_1/while/gru_cell_5/ReluRelu9bidirectional_1/backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
5bidirectional_1/backward_gru_1/while/gru_cell_5/mul_1Mul;bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid:y:02bidirectional_1_backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
5bidirectional_1/backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?é
3bidirectional_1/backward_gru_1/while/gru_cell_5/subSub>bidirectional_1/backward_gru_1/while/gru_cell_5/sub/x:output:0;bidirectional_1/backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë
5bidirectional_1/backward_gru_1/while/gru_cell_5/mul_2Mul7bidirectional_1/backward_gru_1/while/gru_cell_5/sub:z:0Bbidirectional_1/backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
5bidirectional_1/backward_gru_1/while/gru_cell_5/add_3AddV29bidirectional_1/backward_gru_1/while/gru_cell_5/mul_1:z:09bidirectional_1/backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
Ibidirectional_1/backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2bidirectional_1_backward_gru_1_while_placeholder_10bidirectional_1_backward_gru_1_while_placeholder9bidirectional_1/backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒl
*bidirectional_1/backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(bidirectional_1/backward_gru_1/while/addAddV20bidirectional_1_backward_gru_1_while_placeholder3bidirectional_1/backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: n
,bidirectional_1/backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*bidirectional_1/backward_gru_1/while/add_1AddV2Vbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_loop_counter5bidirectional_1/backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: ¶
-bidirectional_1/backward_gru_1/while/IdentityIdentity.bidirectional_1/backward_gru_1/while/add_1:z:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: æ
/bidirectional_1/backward_gru_1/while/Identity_1Identity\bidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_while_maximum_iterations*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¶
/bidirectional_1/backward_gru_1/while/Identity_2Identity,bidirectional_1/backward_gru_1/while/add:z:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: ö
/bidirectional_1/backward_gru_1/while/Identity_3IdentityYbidirectional_1/backward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒÔ
/bidirectional_1/backward_gru_1/while/Identity_4Identity9bidirectional_1/backward_gru_1/while/gru_cell_5/add_3:z:0*^bidirectional_1/backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¾
)bidirectional_1/backward_gru_1/while/NoOpNoOpF^bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpH^bidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp?^bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "¬
Sbidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1Ubidirectional_1_backward_gru_1_while_bidirectional_1_backward_gru_1_strided_slice_1_0"¦
Pbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceRbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"¢
Nbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resourcePbidirectional_1_backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"
Gbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resourceIbidirectional_1_backward_gru_1_while_gru_cell_5_readvariableop_resource_0"g
-bidirectional_1_backward_gru_1_while_identity6bidirectional_1/backward_gru_1/while/Identity:output:0"k
/bidirectional_1_backward_gru_1_while_identity_18bidirectional_1/backward_gru_1/while/Identity_1:output:0"k
/bidirectional_1_backward_gru_1_while_identity_28bidirectional_1/backward_gru_1/while/Identity_2:output:0"k
/bidirectional_1_backward_gru_1_while_identity_38bidirectional_1/backward_gru_1/while/Identity_3:output:0"k
/bidirectional_1_backward_gru_1_while_identity_48bidirectional_1/backward_gru_1/while/Identity_4:output:0"¦
bidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensorbidirectional_1_backward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Ebidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpEbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2
Gbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpGbidirectional_1/backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2
>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp>bidirectional_1/backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Â<
ø
while_body_46463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_5_readvariableop_resource_0:C
1while_gru_cell_5_matmul_readvariableop_resource_0:E
3while_gru_cell_5_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_5_readvariableop_resource:A
/while_gru_cell_5_matmul_readvariableop_resource:C
1while_gru_cell_5_matmul_1_readvariableop_resource:
¢&while/gru_cell_5/MatMul/ReadVariableOp¢(while/gru_cell_5/MatMul_1/ReadVariableOp¢while/gru_cell_5/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Ð

¯
forward_gru_1_while_cond_422858
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_2:
6forward_gru_1_while_less_forward_gru_1_strided_slice_1O
Kforward_gru_1_while_forward_gru_1_while_cond_42285___redundant_placeholder0O
Kforward_gru_1_while_forward_gru_1_while_cond_42285___redundant_placeholder1O
Kforward_gru_1_while_forward_gru_1_while_cond_42285___redundant_placeholder2O
Kforward_gru_1_while_forward_gru_1_while_cond_42285___redundant_placeholder3 
forward_gru_1_while_identity

forward_gru_1/while/LessLessforward_gru_1_while_placeholder6forward_gru_1_while_less_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: g
forward_gru_1/while/IdentityIdentityforward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
[
Û
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46558

inputs4
"gru_cell_5_readvariableop_resource:;
)gru_cell_5_matmul_readvariableop_resource:=
+gru_cell_5_matmul_1_readvariableop_resource:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_5/MatMul/ReadVariableOp¢"gru_cell_5/MatMul_1/ReadVariableOp¢gru_cell_5/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46463*
condR
while_cond_46462*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¼
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸r

;sequential_1_bidirectional_1_forward_gru_1_while_body_40448r
nsequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_loop_counterx
tsequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_maximum_iterations@
<sequential_1_bidirectional_1_forward_gru_1_while_placeholderB
>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_1B
>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_2q
msequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_strided_slice_1_0®
©sequential_1_bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0:n
\sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:p
^sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
=
9sequential_1_bidirectional_1_forward_gru_1_while_identity?
;sequential_1_bidirectional_1_forward_gru_1_while_identity_1?
;sequential_1_bidirectional_1_forward_gru_1_while_identity_2?
;sequential_1_bidirectional_1_forward_gru_1_while_identity_3?
;sequential_1_bidirectional_1_forward_gru_1_while_identity_4o
ksequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_strided_slice_1¬
§sequential_1_bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensore
Ssequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource:l
Zsequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:n
\sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢Qsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢Ssequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp³
bsequential_1/bidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   þ
Tsequential_1/bidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem©sequential_1_bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0<sequential_1_bidirectional_1_forward_gru_1_while_placeholderksequential_1/bidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0à
Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOpUsequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0×
Csequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/unstackUnpackRsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numî
Qsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp\sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0¶
Bsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMulMatMul[sequential_1/bidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Ysequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Csequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAddBiasAddLsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul:product:0Lsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ksequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/splitSplitTsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split/split_dim:output:0Lsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitò
Ssequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp^sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
Dsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1MatMul>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_2[sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Esequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAddNsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1:product:0Lsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Msequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¶
Csequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split_1SplitVNsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/Const:output:0Vsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
?sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/addAddV2Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:0Lsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Å
Csequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/SigmoidSigmoidCsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_1AddV2Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:1Lsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
É
Esequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid_1SigmoidEsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

?sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/mulMulIsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0Lsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_2AddV2Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:2Csequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
@sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReluReluEsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/mul_1MulGsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid:y:0>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
?sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/subSubJsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/sub/x:output:0Gsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/mul_2MulCsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/sub:z:0Nsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Asequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_3AddV2Esequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/mul_1:z:0Esequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ï
Usequential_1/bidirectional_1/forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_1<sequential_1_bidirectional_1_forward_gru_1_while_placeholderEsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒx
6sequential_1/bidirectional_1/forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ý
4sequential_1/bidirectional_1/forward_gru_1/while/addAddV2<sequential_1_bidirectional_1_forward_gru_1_while_placeholder?sequential_1/bidirectional_1/forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: z
8sequential_1/bidirectional_1/forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
6sequential_1/bidirectional_1/forward_gru_1/while/add_1AddV2nsequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_loop_counterAsequential_1/bidirectional_1/forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: Ú
9sequential_1/bidirectional_1/forward_gru_1/while/IdentityIdentity:sequential_1/bidirectional_1/forward_gru_1/while/add_1:z:06^sequential_1/bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
;sequential_1/bidirectional_1/forward_gru_1/while/Identity_1Identitytsequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_maximum_iterations6^sequential_1/bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ú
;sequential_1/bidirectional_1/forward_gru_1/while/Identity_2Identity8sequential_1/bidirectional_1/forward_gru_1/while/add:z:06^sequential_1/bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
;sequential_1/bidirectional_1/forward_gru_1/while/Identity_3Identityesequential_1/bidirectional_1/forward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^sequential_1/bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒø
;sequential_1/bidirectional_1/forward_gru_1/while/Identity_4IdentityEsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/add_3:z:06^sequential_1/bidirectional_1/forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
5sequential_1/bidirectional_1/forward_gru_1/while/NoOpNoOpR^sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpT^sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpK^sequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "¾
\sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource^sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"º
Zsequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource\sequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"¬
Ssequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resourceUsequential_1_bidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0"
9sequential_1_bidirectional_1_forward_gru_1_while_identityBsequential_1/bidirectional_1/forward_gru_1/while/Identity:output:0"
;sequential_1_bidirectional_1_forward_gru_1_while_identity_1Dsequential_1/bidirectional_1/forward_gru_1/while/Identity_1:output:0"
;sequential_1_bidirectional_1_forward_gru_1_while_identity_2Dsequential_1/bidirectional_1/forward_gru_1/while/Identity_2:output:0"
;sequential_1_bidirectional_1_forward_gru_1_while_identity_3Dsequential_1/bidirectional_1/forward_gru_1/while/Identity_3:output:0"
;sequential_1_bidirectional_1_forward_gru_1_while_identity_4Dsequential_1/bidirectional_1/forward_gru_1/while/Identity_4:output:0"Ü
ksequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_strided_slice_1msequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_strided_slice_1_0"Ö
§sequential_1_bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor©sequential_1_bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2¦
Qsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpQsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2ª
Ssequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpSsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2
Jsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOpJsequential_1/bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
¤O
º
backward_gru_1_while_body_44878:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_29
5backward_gru_1_while_backward_gru_1_strided_slice_1_0u
qbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0K
9backward_gru_1_while_gru_cell_5_readvariableop_resource_0:R
@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:T
Bbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
!
backward_gru_1_while_identity#
backward_gru_1_while_identity_1#
backward_gru_1_while_identity_2#
backward_gru_1_while_identity_3#
backward_gru_1_while_identity_47
3backward_gru_1_while_backward_gru_1_strided_slice_1s
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorI
7backward_gru_1_while_gru_cell_5_readvariableop_resource:P
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:R
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢.backward_gru_1/while/gru_cell_5/ReadVariableOp
Fbackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0 backward_gru_1_while_placeholderObackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp9backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
'backward_gru_1/while/gru_cell_5/unstackUnpack6backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¶
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0â
&backward_gru_1/while/gru_cell_5/MatMulMatMul?backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'backward_gru_1/while/gru_cell_5/BiasAddBiasAdd0backward_gru_1/while/gru_cell_5/MatMul:product:00backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
%backward_gru_1/while/gru_cell_5/splitSplit8backward_gru_1/while/gru_cell_5/split/split_dim:output:00backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0É
(backward_gru_1/while/gru_cell_5/MatMul_1MatMul"backward_gru_1_while_placeholder_2?backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAdd2backward_gru_1/while/gru_cell_5/MatMul_1:product:00backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
%backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ|
1backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
'backward_gru_1/while/gru_cell_5/split_1SplitV2backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0.backward_gru_1/while/gru_cell_5/Const:output:0:backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÀ
#backward_gru_1/while/gru_cell_5/addAddV2.backward_gru_1/while/gru_cell_5/split:output:00backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru_1/while/gru_cell_5/SigmoidSigmoid'backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
%backward_gru_1/while/gru_cell_5/add_1AddV2.backward_gru_1/while/gru_cell_5/split:output:10backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid)backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
#backward_gru_1/while/gru_cell_5/mulMul-backward_gru_1/while/gru_cell_5/Sigmoid_1:y:00backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
%backward_gru_1/while/gru_cell_5/add_2AddV2.backward_gru_1/while/gru_cell_5/split:output:2'backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$backward_gru_1/while/gru_cell_5/ReluRelu)backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
%backward_gru_1/while/gru_cell_5/mul_1Mul+backward_gru_1/while/gru_cell_5/Sigmoid:y:0"backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
%backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
#backward_gru_1/while/gru_cell_5/subSub.backward_gru_1/while/gru_cell_5/sub/x:output:0+backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
%backward_gru_1/while/gru_cell_5/mul_2Mul'backward_gru_1/while/gru_cell_5/sub:z:02backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
%backward_gru_1/while/gru_cell_5/add_3AddV2)backward_gru_1/while/gru_cell_5/mul_1:z:0)backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"backward_gru_1_while_placeholder_1 backward_gru_1_while_placeholder)backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru_1/while/addAddV2 backward_gru_1_while_placeholder#backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ^
backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
backward_gru_1/while/add_1AddV26backward_gru_1_while_backward_gru_1_while_loop_counter%backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/add_1:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¦
backward_gru_1/while/Identity_1Identity<backward_gru_1_while_backward_gru_1_while_maximum_iterations^backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
backward_gru_1/while/Identity_2Identitybackward_gru_1/while/add:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: Æ
backward_gru_1/while/Identity_3IdentityIbackward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
backward_gru_1/while/Identity_4Identity)backward_gru_1/while/gru_cell_5/add_3:z:0^backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
backward_gru_1/while/NoOpNoOp6^backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp8^backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp/^backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3backward_gru_1_while_backward_gru_1_strided_slice_15backward_gru_1_while_backward_gru_1_strided_slice_1_0"
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"t
7backward_gru_1_while_gru_cell_5_readvariableop_resource9backward_gru_1_while_gru_cell_5_readvariableop_resource_0"G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0"K
backward_gru_1_while_identity_1(backward_gru_1/while/Identity_1:output:0"K
backward_gru_1_while_identity_2(backward_gru_1/while/Identity_2:output:0"K
backward_gru_1_while_identity_3(backward_gru_1/while/Identity_3:output:0"K
backward_gru_1_while_identity_4(backward_gru_1/while/Identity_4:output:0"ä
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2n
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2r
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2`
.backward_gru_1/while/gru_cell_5/ReadVariableOp.backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ÉA
Ö
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41423

inputs"
gru_cell_5_41341:"
gru_cell_5_41343:"
gru_cell_5_41345:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_5/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÀ
"gru_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_5_41341gru_cell_5_41343gru_cell_5_41345*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41299n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ó
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_5_41341gru_cell_5_41343gru_cell_5_41345*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_41353*
condR
while_cond_41352*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    £
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_5_41343*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È
NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_5/StatefulPartitionedCall"gru_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ú
G__inference_sequential_1_layer_call_and_return_conditional_losses_43588

inputsR
@bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resource:Y
Gbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource:[
Ibidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
S
Abidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resource:Z
Hbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource:\
Jbidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity¢?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢Abidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢$bidirectional_1/backward_gru_1/while¢>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢#bidirectional_1/forward_gru_1/while¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOpY
#bidirectional_1/forward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:{
1bidirectional_1/forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3bidirectional_1/forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3bidirectional_1/forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+bidirectional_1/forward_gru_1/strided_sliceStridedSlice,bidirectional_1/forward_gru_1/Shape:output:0:bidirectional_1/forward_gru_1/strided_slice/stack:output:0<bidirectional_1/forward_gru_1/strided_slice/stack_1:output:0<bidirectional_1/forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,bidirectional_1/forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Í
*bidirectional_1/forward_gru_1/zeros/packedPack4bidirectional_1/forward_gru_1/strided_slice:output:05bidirectional_1/forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:n
)bidirectional_1/forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Æ
#bidirectional_1/forward_gru_1/zerosFill3bidirectional_1/forward_gru_1/zeros/packed:output:02bidirectional_1/forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

,bidirectional_1/forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ©
'bidirectional_1/forward_gru_1/transpose	Transposeinputs5bidirectional_1/forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%bidirectional_1/forward_gru_1/Shape_1Shape+bidirectional_1/forward_gru_1/transpose:y:0*
T0*
_output_shapes
:}
3bidirectional_1/forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5bidirectional_1/forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_1/forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-bidirectional_1/forward_gru_1/strided_slice_1StridedSlice.bidirectional_1/forward_gru_1/Shape_1:output:0<bidirectional_1/forward_gru_1/strided_slice_1/stack:output:0>bidirectional_1/forward_gru_1/strided_slice_1/stack_1:output:0>bidirectional_1/forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
9bidirectional_1/forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+bidirectional_1/forward_gru_1/TensorArrayV2TensorListReserveBbidirectional_1/forward_gru_1/TensorArrayV2/element_shape:output:06bidirectional_1/forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ¤
Sbidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   º
Ebidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor+bidirectional_1/forward_gru_1/transpose:y:0\bidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ}
3bidirectional_1/forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5bidirectional_1/forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5bidirectional_1/forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ÿ
-bidirectional_1/forward_gru_1/strided_slice_2StridedSlice+bidirectional_1/forward_gru_1/transpose:y:0<bidirectional_1/forward_gru_1/strided_slice_2/stack:output:0>bidirectional_1/forward_gru_1/strided_slice_2/stack_1:output:0>bidirectional_1/forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¸
7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOp@bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0±
0bidirectional_1/forward_gru_1/gru_cell_4/unstackUnpack?bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÆ
>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOpGbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ë
/bidirectional_1/forward_gru_1/gru_cell_4/MatMulMatMul6bidirectional_1/forward_gru_1/strided_slice_2:output:0Fbidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
0bidirectional_1/forward_gru_1/gru_cell_4/BiasAddBiasAdd9bidirectional_1/forward_gru_1/gru_cell_4/MatMul:product:09bidirectional_1/forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8bidirectional_1/forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
.bidirectional_1/forward_gru_1/gru_cell_4/splitSplitAbidirectional_1/forward_gru_1/gru_cell_4/split/split_dim:output:09bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÊ
@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpIbidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0å
1bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1MatMul,bidirectional_1/forward_gru_1/zeros:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
2bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd_1BiasAdd;bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1:product:09bidirectional_1/forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.bidirectional_1/forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
:bidirectional_1/forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
0bidirectional_1/forward_gru_1/gru_cell_4/split_1SplitV;bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd_1:output:07bidirectional_1/forward_gru_1/gru_cell_4/Const:output:0Cbidirectional_1/forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÛ
,bidirectional_1/forward_gru_1/gru_cell_4/addAddV27bidirectional_1/forward_gru_1/gru_cell_4/split:output:09bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

0bidirectional_1/forward_gru_1/gru_cell_4/SigmoidSigmoid0bidirectional_1/forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
.bidirectional_1/forward_gru_1/gru_cell_4/add_1AddV27bidirectional_1/forward_gru_1/gru_cell_4/split:output:19bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid2bidirectional_1/forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ø
,bidirectional_1/forward_gru_1/gru_cell_4/mulMul6bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid_1:y:09bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô
.bidirectional_1/forward_gru_1/gru_cell_4/add_2AddV27bidirectional_1/forward_gru_1/gru_cell_4/split:output:20bidirectional_1/forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-bidirectional_1/forward_gru_1/gru_cell_4/ReluRelu2bidirectional_1/forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ë
.bidirectional_1/forward_gru_1/gru_cell_4/mul_1Mul4bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid:y:0,bidirectional_1/forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
.bidirectional_1/forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ô
,bidirectional_1/forward_gru_1/gru_cell_4/subSub7bidirectional_1/forward_gru_1/gru_cell_4/sub/x:output:04bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
.bidirectional_1/forward_gru_1/gru_cell_4/mul_2Mul0bidirectional_1/forward_gru_1/gru_cell_4/sub:z:0;bidirectional_1/forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ñ
.bidirectional_1/forward_gru_1/gru_cell_4/add_3AddV22bidirectional_1/forward_gru_1/gru_cell_4/mul_1:z:02bidirectional_1/forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;bidirectional_1/forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
-bidirectional_1/forward_gru_1/TensorArrayV2_1TensorListReserveDbidirectional_1/forward_gru_1/TensorArrayV2_1/element_shape:output:06bidirectional_1/forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
"bidirectional_1/forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
6bidirectional_1/forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿr
0bidirectional_1/forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¿
#bidirectional_1/forward_gru_1/whileWhile9bidirectional_1/forward_gru_1/while/loop_counter:output:0?bidirectional_1/forward_gru_1/while/maximum_iterations:output:0+bidirectional_1/forward_gru_1/time:output:06bidirectional_1/forward_gru_1/TensorArrayV2_1:handle:0,bidirectional_1/forward_gru_1/zeros:output:06bidirectional_1/forward_gru_1/strided_slice_1:output:0Ubidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resourceGbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resourceIbidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *:
body2R0
.bidirectional_1_forward_gru_1_while_body_43320*:
cond2R0
.bidirectional_1_forward_gru_1_while_cond_43319*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
Nbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
@bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStack,bidirectional_1/forward_gru_1/while:output:3Wbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
3bidirectional_1/forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
5bidirectional_1/forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5bidirectional_1/forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
-bidirectional_1/forward_gru_1/strided_slice_3StridedSliceIbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0<bidirectional_1/forward_gru_1/strided_slice_3/stack:output:0>bidirectional_1/forward_gru_1/strided_slice_3/stack_1:output:0>bidirectional_1/forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
.bidirectional_1/forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ð
)bidirectional_1/forward_gru_1/transpose_1	TransposeIbidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:07bidirectional_1/forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
%bidirectional_1/forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Z
$bidirectional_1/backward_gru_1/ShapeShapeinputs*
T0*
_output_shapes
:|
2bidirectional_1/backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4bidirectional_1/backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4bidirectional_1/backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,bidirectional_1/backward_gru_1/strided_sliceStridedSlice-bidirectional_1/backward_gru_1/Shape:output:0;bidirectional_1/backward_gru_1/strided_slice/stack:output:0=bidirectional_1/backward_gru_1/strided_slice/stack_1:output:0=bidirectional_1/backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-bidirectional_1/backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Ð
+bidirectional_1/backward_gru_1/zeros/packedPack5bidirectional_1/backward_gru_1/strided_slice:output:06bidirectional_1/backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:o
*bidirectional_1/backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    É
$bidirectional_1/backward_gru_1/zerosFill4bidirectional_1/backward_gru_1/zeros/packed:output:03bidirectional_1/backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-bidirectional_1/backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
(bidirectional_1/backward_gru_1/transpose	Transposeinputs6bidirectional_1/backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&bidirectional_1/backward_gru_1/Shape_1Shape,bidirectional_1/backward_gru_1/transpose:y:0*
T0*
_output_shapes
:~
4bidirectional_1/backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6bidirectional_1/backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_1/backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.bidirectional_1/backward_gru_1/strided_slice_1StridedSlice/bidirectional_1/backward_gru_1/Shape_1:output:0=bidirectional_1/backward_gru_1/strided_slice_1/stack:output:0?bidirectional_1/backward_gru_1/strided_slice_1/stack_1:output:0?bidirectional_1/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:bidirectional_1/backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,bidirectional_1/backward_gru_1/TensorArrayV2TensorListReserveCbidirectional_1/backward_gru_1/TensorArrayV2/element_shape:output:07bidirectional_1/backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒw
-bidirectional_1/backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ñ
(bidirectional_1/backward_gru_1/ReverseV2	ReverseV2,bidirectional_1/backward_gru_1/transpose:y:06bidirectional_1/backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Tbidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
Fbidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor1bidirectional_1/backward_gru_1/ReverseV2:output:0]bidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ~
4bidirectional_1/backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6bidirectional_1/backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6bidirectional_1/backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
.bidirectional_1/backward_gru_1/strided_slice_2StridedSlice,bidirectional_1/backward_gru_1/transpose:y:0=bidirectional_1/backward_gru_1/strided_slice_2/stack:output:0?bidirectional_1/backward_gru_1/strided_slice_2/stack_1:output:0?bidirectional_1/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskº
8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOpAbidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0³
1bidirectional_1/backward_gru_1/gru_cell_5/unstackUnpack@bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÈ
?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOpHbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0î
0bidirectional_1/backward_gru_1/gru_cell_5/MatMulMatMul7bidirectional_1/backward_gru_1/strided_slice_2:output:0Gbidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
1bidirectional_1/backward_gru_1/gru_cell_5/BiasAddBiasAdd:bidirectional_1/backward_gru_1/gru_cell_5/MatMul:product:0:bidirectional_1/backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9bidirectional_1/backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¡
/bidirectional_1/backward_gru_1/gru_cell_5/splitSplitBbidirectional_1/backward_gru_1/gru_cell_5/split/split_dim:output:0:bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÌ
Abidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpJbidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0è
2bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1MatMul-bidirectional_1/backward_gru_1/zeros:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
3bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd_1BiasAdd<bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1:product:0:bidirectional_1/backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/bidirectional_1/backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
;bidirectional_1/backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿî
1bidirectional_1/backward_gru_1/gru_cell_5/split_1SplitV<bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd_1:output:08bidirectional_1/backward_gru_1/gru_cell_5/Const:output:0Dbidirectional_1/backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÞ
-bidirectional_1/backward_gru_1/gru_cell_5/addAddV28bidirectional_1/backward_gru_1/gru_cell_5/split:output:0:bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
1bidirectional_1/backward_gru_1/gru_cell_5/SigmoidSigmoid1bidirectional_1/backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à
/bidirectional_1/backward_gru_1/gru_cell_5/add_1AddV28bidirectional_1/backward_gru_1/gru_cell_5/split:output:1:bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
3bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid3bidirectional_1/backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Û
-bidirectional_1/backward_gru_1/gru_cell_5/mulMul7bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid_1:y:0:bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
/bidirectional_1/backward_gru_1/gru_cell_5/add_2AddV28bidirectional_1/backward_gru_1/gru_cell_5/split:output:21bidirectional_1/backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

.bidirectional_1/backward_gru_1/gru_cell_5/ReluRelu3bidirectional_1/backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Î
/bidirectional_1/backward_gru_1/gru_cell_5/mul_1Mul5bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid:y:0-bidirectional_1/backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
t
/bidirectional_1/backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?×
-bidirectional_1/backward_gru_1/gru_cell_5/subSub8bidirectional_1/backward_gru_1/gru_cell_5/sub/x:output:05bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ù
/bidirectional_1/backward_gru_1/gru_cell_5/mul_2Mul1bidirectional_1/backward_gru_1/gru_cell_5/sub:z:0<bidirectional_1/backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô
/bidirectional_1/backward_gru_1/gru_cell_5/add_3AddV23bidirectional_1/backward_gru_1/gru_cell_5/mul_1:z:03bidirectional_1/backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<bidirectional_1/backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
.bidirectional_1/backward_gru_1/TensorArrayV2_1TensorListReserveEbidirectional_1/backward_gru_1/TensorArrayV2_1/element_shape:output:07bidirectional_1/backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
#bidirectional_1/backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
7bidirectional_1/backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1bidirectional_1/backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ì
$bidirectional_1/backward_gru_1/whileWhile:bidirectional_1/backward_gru_1/while/loop_counter:output:0@bidirectional_1/backward_gru_1/while/maximum_iterations:output:0,bidirectional_1/backward_gru_1/time:output:07bidirectional_1/backward_gru_1/TensorArrayV2_1:handle:0-bidirectional_1/backward_gru_1/zeros:output:07bidirectional_1/backward_gru_1/strided_slice_1:output:0Vbidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Abidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resourceHbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resourceJbidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *;
body3R1
/bidirectional_1_backward_gru_1_while_body_43471*;
cond3R1
/bidirectional_1_backward_gru_1_while_cond_43470*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations  
Obidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
Abidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStack-bidirectional_1/backward_gru_1/while:output:3Xbidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
4bidirectional_1/backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
6bidirectional_1/backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6bidirectional_1/backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
.bidirectional_1/backward_gru_1/strided_slice_3StridedSliceJbidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0=bidirectional_1/backward_gru_1/strided_slice_3/stack:output:0?bidirectional_1/backward_gru_1/strided_slice_3/stack_1:output:0?bidirectional_1/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
/bidirectional_1/backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ó
*bidirectional_1/backward_gru_1/transpose_1	TransposeJbidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:08bidirectional_1/backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
&bidirectional_1/backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ô
bidirectional_1/concatConcatV26bidirectional_1/forward_gru_1/strided_slice_3:output:07bidirectional_1/backward_gru_1/strided_slice_3:output:0$bidirectional_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMulbidirectional_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpGbidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Û
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpHbidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp@^bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpB^bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp9^bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp%^bidirectional_1/backward_gru_1/while?^bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpA^bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp8^bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp$^bidirectional_1/forward_gru_1/while^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2
?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp?bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2
Abidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpAbidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2t
8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp8bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2L
$bidirectional_1/backward_gru_1/while$bidirectional_1/backward_gru_1/while2
>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp>bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2
@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp@bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2r
7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp7bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2J
#bidirectional_1/forward_gru_1/while#bidirectional_1/forward_gru_1/while2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
.__inference_backward_gru_1_layer_call_fn_46053
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Â<
ø
while_body_46624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_5_readvariableop_resource_0:C
1while_gru_cell_5_matmul_readvariableop_resource_0:E
3while_gru_cell_5_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_5_readvariableop_resource:A
/while_gru_cell_5_matmul_readvariableop_resource:C
1while_gru_cell_5_matmul_1_readvariableop_resource:
¢&while/gru_cell_5/MatMul/ReadVariableOp¢(while/gru_cell_5/MatMul_1/ReadVariableOp¢while/gru_cell_5/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
­O
º
backward_gru_1_while_body_44560:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_29
5backward_gru_1_while_backward_gru_1_strided_slice_1_0u
qbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0K
9backward_gru_1_while_gru_cell_5_readvariableop_resource_0:R
@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:T
Bbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
!
backward_gru_1_while_identity#
backward_gru_1_while_identity_1#
backward_gru_1_while_identity_2#
backward_gru_1_while_identity_3#
backward_gru_1_while_identity_47
3backward_gru_1_while_backward_gru_1_strided_slice_1s
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorI
7backward_gru_1_while_gru_cell_5_readvariableop_resource:P
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:R
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢.backward_gru_1/while/gru_cell_5/ReadVariableOp
Fbackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿú
8backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0 backward_gru_1_while_placeholderObackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp9backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
'backward_gru_1/while/gru_cell_5/unstackUnpack6backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¶
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0â
&backward_gru_1/while/gru_cell_5/MatMulMatMul?backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'backward_gru_1/while/gru_cell_5/BiasAddBiasAdd0backward_gru_1/while/gru_cell_5/MatMul:product:00backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
%backward_gru_1/while/gru_cell_5/splitSplit8backward_gru_1/while/gru_cell_5/split/split_dim:output:00backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0É
(backward_gru_1/while/gru_cell_5/MatMul_1MatMul"backward_gru_1_while_placeholder_2?backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAdd2backward_gru_1/while/gru_cell_5/MatMul_1:product:00backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
%backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ|
1backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
'backward_gru_1/while/gru_cell_5/split_1SplitV2backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0.backward_gru_1/while/gru_cell_5/Const:output:0:backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÀ
#backward_gru_1/while/gru_cell_5/addAddV2.backward_gru_1/while/gru_cell_5/split:output:00backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru_1/while/gru_cell_5/SigmoidSigmoid'backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
%backward_gru_1/while/gru_cell_5/add_1AddV2.backward_gru_1/while/gru_cell_5/split:output:10backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid)backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
#backward_gru_1/while/gru_cell_5/mulMul-backward_gru_1/while/gru_cell_5/Sigmoid_1:y:00backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
%backward_gru_1/while/gru_cell_5/add_2AddV2.backward_gru_1/while/gru_cell_5/split:output:2'backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$backward_gru_1/while/gru_cell_5/ReluRelu)backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
%backward_gru_1/while/gru_cell_5/mul_1Mul+backward_gru_1/while/gru_cell_5/Sigmoid:y:0"backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
%backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
#backward_gru_1/while/gru_cell_5/subSub.backward_gru_1/while/gru_cell_5/sub/x:output:0+backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
%backward_gru_1/while/gru_cell_5/mul_2Mul'backward_gru_1/while/gru_cell_5/sub:z:02backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
%backward_gru_1/while/gru_cell_5/add_3AddV2)backward_gru_1/while/gru_cell_5/mul_1:z:0)backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"backward_gru_1_while_placeholder_1 backward_gru_1_while_placeholder)backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru_1/while/addAddV2 backward_gru_1_while_placeholder#backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ^
backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
backward_gru_1/while/add_1AddV26backward_gru_1_while_backward_gru_1_while_loop_counter%backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/add_1:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¦
backward_gru_1/while/Identity_1Identity<backward_gru_1_while_backward_gru_1_while_maximum_iterations^backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
backward_gru_1/while/Identity_2Identitybackward_gru_1/while/add:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: Æ
backward_gru_1/while/Identity_3IdentityIbackward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
backward_gru_1/while/Identity_4Identity)backward_gru_1/while/gru_cell_5/add_3:z:0^backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
backward_gru_1/while/NoOpNoOp6^backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp8^backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp/^backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3backward_gru_1_while_backward_gru_1_strided_slice_15backward_gru_1_while_backward_gru_1_strided_slice_1_0"
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"t
7backward_gru_1_while_gru_cell_5_readvariableop_resource9backward_gru_1_while_gru_cell_5_readvariableop_resource_0"G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0"K
backward_gru_1_while_identity_1(backward_gru_1/while/Identity_1:output:0"K
backward_gru_1_while_identity_2(backward_gru_1/while/Identity_2:output:0"K
backward_gru_1_while_identity_3(backward_gru_1/while/Identity_3:output:0"K
backward_gru_1_while_identity_4(backward_gru_1/while/Identity_4:output:0"ä
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2n
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2r
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2`
.backward_gru_1/while/gru_cell_5/ReadVariableOp.backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_46623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46623___redundant_placeholder03
/while_while_cond_46623___redundant_placeholder13
/while_while_cond_46623___redundant_placeholder23
/while_while_cond_46623___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
Â<
ø
while_body_41502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_4_readvariableop_resource_0:C
1while_gru_cell_4_matmul_readvariableop_resource_0:E
3while_gru_cell_4_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_4_readvariableop_resource:A
/while_gru_cell_4_matmul_readvariableop_resource:C
1while_gru_cell_4_matmul_1_readvariableop_resource:
¢&while/gru_cell_4/MatMul/ReadVariableOp¢(while/gru_cell_4/MatMul_1/ReadVariableOp¢while/gru_cell_4/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
×,
þ
G__inference_sequential_1_layer_call_and_return_conditional_losses_43149
bidirectional_1_input'
bidirectional_1_43113:'
bidirectional_1_43115:'
bidirectional_1_43117:
'
bidirectional_1_43119:'
bidirectional_1_43121:'
bidirectional_1_43123:

dense_2_43126:
dense_2_43128:
dense_3_43131:
dense_3_43133:
identity¢'bidirectional_1/StatefulPartitionedCall¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallü
'bidirectional_1/StatefulPartitionedCallStatefulPartitionedCallbidirectional_1_inputbidirectional_1_43113bidirectional_1_43115bidirectional_1_43117bidirectional_1_43119bidirectional_1_43121bidirectional_1_43123*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42540
dense_2/StatefulPartitionedCallStatefulPartitionedCall0bidirectional_1/StatefulPartitionedCall:output:0dense_2_43126dense_2_43128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42565
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_43131dense_3_43133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42582§
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_43115*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_1_43121*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
NoOpNoOp(^bidirectional_1/StatefulPartitionedCallS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2R
'bidirectional_1/StatefulPartitionedCall'bidirectional_1/StatefulPartitionedCall2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:b ^
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namebidirectional_1_input
åZ
Ý
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46236
inputs_04
"gru_cell_5_readvariableop_resource:;
)gru_cell_5_matmul_readvariableop_resource:=
+gru_cell_5_matmul_1_readvariableop_resource:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_5/MatMul/ReadVariableOp¢"gru_cell_5/MatMul_1/ReadVariableOp¢gru_cell_5/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   å
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_46141*
condR
while_cond_46140*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¼
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ú£
Ä
!__inference__traced_restore_47256
file_prefix1
assignvariableop_dense_2_kernel:-
assignvariableop_1_dense_2_bias:3
!assignvariableop_2_dense_3_kernel:-
assignvariableop_3_dense_3_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: T
Bassignvariableop_9_bidirectional_1_forward_gru_1_gru_cell_4_kernel:_
Massignvariableop_10_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel:
S
Aassignvariableop_11_bidirectional_1_forward_gru_1_gru_cell_4_bias:V
Dassignvariableop_12_bidirectional_1_backward_gru_1_gru_cell_5_kernel:`
Nassignvariableop_13_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel:
T
Bassignvariableop_14_bidirectional_1_backward_gru_1_gru_cell_5_bias:#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: ;
)assignvariableop_19_adam_dense_2_kernel_m:5
'assignvariableop_20_adam_dense_2_bias_m:;
)assignvariableop_21_adam_dense_3_kernel_m:5
'assignvariableop_22_adam_dense_3_bias_m:\
Jassignvariableop_23_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_m:f
Tassignvariableop_24_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_m:
Z
Hassignvariableop_25_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_m:]
Kassignvariableop_26_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_m:g
Uassignvariableop_27_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_m:
[
Iassignvariableop_28_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_m:;
)assignvariableop_29_adam_dense_2_kernel_v:5
'assignvariableop_30_adam_dense_2_bias_v:;
)assignvariableop_31_adam_dense_3_kernel_v:5
'assignvariableop_32_adam_dense_3_bias_v:\
Jassignvariableop_33_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_v:f
Tassignvariableop_34_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_v:
Z
Hassignvariableop_35_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_v:]
Kassignvariableop_36_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_v:g
Uassignvariableop_37_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_v:
[
Iassignvariableop_38_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_v:
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueúB÷(B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_9AssignVariableOpBassignvariableop_9_bidirectional_1_forward_gru_1_gru_cell_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_10AssignVariableOpMassignvariableop_10_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_11AssignVariableOpAassignvariableop_11_bidirectional_1_forward_gru_1_gru_cell_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_12AssignVariableOpDassignvariableop_12_bidirectional_1_backward_gru_1_gru_cell_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_13AssignVariableOpNassignvariableop_13_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_14AssignVariableOpBassignvariableop_14_bidirectional_1_backward_gru_1_gru_cell_5_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_23AssignVariableOpJassignvariableop_23_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_24AssignVariableOpTassignvariableop_24_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_25AssignVariableOpHassignvariableop_25_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_26AssignVariableOpKassignvariableop_26_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_27AssignVariableOpUassignvariableop_27_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_28AssignVariableOpIassignvariableop_28_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_33AssignVariableOpJassignvariableop_33_adam_bidirectional_1_forward_gru_1_gru_cell_4_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_34AssignVariableOpTassignvariableop_34_adam_bidirectional_1_forward_gru_1_gru_cell_4_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_35AssignVariableOpHassignvariableop_35_adam_bidirectional_1_forward_gru_1_gru_cell_4_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_36AssignVariableOpKassignvariableop_36_adam_bidirectional_1_backward_gru_1_gru_cell_5_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Æ
AssignVariableOp_37AssignVariableOpUassignvariableop_37_adam_bidirectional_1_backward_gru_1_gru_cell_5_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_38AssignVariableOpIassignvariableop_38_adam_bidirectional_1_backward_gru_1_gru_cell_5_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ©
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
÷X
Û
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45707
inputs_04
"gru_cell_4_readvariableop_resource:;
)gru_cell_4_matmul_readvariableop_resource:=
+gru_cell_4_matmul_1_readvariableop_resource:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_4/MatMul/ReadVariableOp¢"gru_cell_4/MatMul_1/ReadVariableOp¢gru_cell_4/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_45612*
condR
while_cond_45611*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    »
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ð

¯
forward_gru_1_while_cond_444088
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_2:
6forward_gru_1_while_less_forward_gru_1_strided_slice_1O
Kforward_gru_1_while_forward_gru_1_while_cond_44408___redundant_placeholder0O
Kforward_gru_1_while_forward_gru_1_while_cond_44408___redundant_placeholder1O
Kforward_gru_1_while_forward_gru_1_while_cond_44408___redundant_placeholder2O
Kforward_gru_1_while_forward_gru_1_while_cond_44408___redundant_placeholder3 
forward_gru_1_while_identity

forward_gru_1/while/LessLessforward_gru_1_while_placeholder6forward_gru_1_while_less_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: g
forward_gru_1/while/IdentityIdentityforward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
¨
¼
.__inference_backward_gru_1_layer_call_fn_46075

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_40792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_40792___redundant_placeholder03
/while_while_cond_40792___redundant_placeholder13
/while_while_cond_40792___redundant_placeholder23
/while_while_cond_40792___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
¤O
º
backward_gru_1_while_body_42878:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_29
5backward_gru_1_while_backward_gru_1_strided_slice_1_0u
qbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0K
9backward_gru_1_while_gru_cell_5_readvariableop_resource_0:R
@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:T
Bbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
!
backward_gru_1_while_identity#
backward_gru_1_while_identity_1#
backward_gru_1_while_identity_2#
backward_gru_1_while_identity_3#
backward_gru_1_while_identity_47
3backward_gru_1_while_backward_gru_1_strided_slice_1s
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorI
7backward_gru_1_while_gru_cell_5_readvariableop_resource:P
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource:R
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
¢5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp¢7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp¢.backward_gru_1/while/gru_cell_5/ReadVariableOp
Fbackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
8backward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0 backward_gru_1_while_placeholderObackward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¨
.backward_gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp9backward_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
'backward_gru_1/while/gru_cell_5/unstackUnpack6backward_gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¶
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0â
&backward_gru_1/while/gru_cell_5/MatMulMatMul?backward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0=backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
'backward_gru_1/while/gru_cell_5/BiasAddBiasAdd0backward_gru_1/while/gru_cell_5/MatMul:product:00backward_gru_1/while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
/backward_gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
%backward_gru_1/while/gru_cell_5/splitSplit8backward_gru_1/while/gru_cell_5/split/split_dim:output:00backward_gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0É
(backward_gru_1/while/gru_cell_5/MatMul_1MatMul"backward_gru_1_while_placeholder_2?backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
)backward_gru_1/while/gru_cell_5/BiasAdd_1BiasAdd2backward_gru_1/while/gru_cell_5/MatMul_1:product:00backward_gru_1/while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
%backward_gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ|
1backward_gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
'backward_gru_1/while/gru_cell_5/split_1SplitV2backward_gru_1/while/gru_cell_5/BiasAdd_1:output:0.backward_gru_1/while/gru_cell_5/Const:output:0:backward_gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÀ
#backward_gru_1/while/gru_cell_5/addAddV2.backward_gru_1/while/gru_cell_5/split:output:00backward_gru_1/while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru_1/while/gru_cell_5/SigmoidSigmoid'backward_gru_1/while/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
%backward_gru_1/while/gru_cell_5/add_1AddV2.backward_gru_1/while/gru_cell_5/split:output:10backward_gru_1/while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)backward_gru_1/while/gru_cell_5/Sigmoid_1Sigmoid)backward_gru_1/while/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
#backward_gru_1/while/gru_cell_5/mulMul-backward_gru_1/while/gru_cell_5/Sigmoid_1:y:00backward_gru_1/while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
%backward_gru_1/while/gru_cell_5/add_2AddV2.backward_gru_1/while/gru_cell_5/split:output:2'backward_gru_1/while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$backward_gru_1/while/gru_cell_5/ReluRelu)backward_gru_1/while/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
%backward_gru_1/while/gru_cell_5/mul_1Mul+backward_gru_1/while/gru_cell_5/Sigmoid:y:0"backward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
%backward_gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
#backward_gru_1/while/gru_cell_5/subSub.backward_gru_1/while/gru_cell_5/sub/x:output:0+backward_gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
%backward_gru_1/while/gru_cell_5/mul_2Mul'backward_gru_1/while/gru_cell_5/sub:z:02backward_gru_1/while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
%backward_gru_1/while/gru_cell_5/add_3AddV2)backward_gru_1/while/gru_cell_5/mul_1:z:0)backward_gru_1/while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9backward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"backward_gru_1_while_placeholder_1 backward_gru_1_while_placeholder)backward_gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
backward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru_1/while/addAddV2 backward_gru_1_while_placeholder#backward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ^
backward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
backward_gru_1/while/add_1AddV26backward_gru_1_while_backward_gru_1_while_loop_counter%backward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/add_1:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: ¦
backward_gru_1/while/Identity_1Identity<backward_gru_1_while_backward_gru_1_while_maximum_iterations^backward_gru_1/while/NoOp*
T0*
_output_shapes
: 
backward_gru_1/while/Identity_2Identitybackward_gru_1/while/add:z:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: Æ
backward_gru_1/while/Identity_3IdentityIbackward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
backward_gru_1/while/Identity_4Identity)backward_gru_1/while/gru_cell_5/add_3:z:0^backward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
backward_gru_1/while/NoOpNoOp6^backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp8^backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp/^backward_gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3backward_gru_1_while_backward_gru_1_strided_slice_15backward_gru_1_while_backward_gru_1_strided_slice_1_0"
@backward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceBbackward_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"
>backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource@backward_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"t
7backward_gru_1_while_gru_cell_5_readvariableop_resource9backward_gru_1_while_gru_cell_5_readvariableop_resource_0"G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0"K
backward_gru_1_while_identity_1(backward_gru_1/while/Identity_1:output:0"K
backward_gru_1_while_identity_2(backward_gru_1/while/Identity_2:output:0"K
backward_gru_1_while_identity_3(backward_gru_1/while/Identity_3:output:0"K
backward_gru_1_while_identity_4(backward_gru_1/while/Identity_4:output:0"ä
obackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensorqbackward_gru_1_while_tensorarrayv2read_tensorlistgetitem_backward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2n
5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp5backward_gru_1/while/gru_cell_5/MatMul/ReadVariableOp2r
7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp7backward_gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2`
.backward_gru_1/while/gru_cell_5/ReadVariableOp.backward_gru_1/while/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 

¾
.__inference_backward_gru_1_layer_call_fn_46042
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41227o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Õ
¥
while_cond_46140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46140___redundant_placeholder03
/while_while_cond_46140___redundant_placeholder13
/while_while_cond_46140___redundant_placeholder23
/while_while_cond_46140___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
[
Û
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41765

inputs4
"gru_cell_5_readvariableop_resource:;
)gru_cell_5_matmul_readvariableop_resource:=
+gru_cell_5_matmul_1_readvariableop_resource:

identity¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_5/MatMul/ReadVariableOp¢"gru_cell_5/MatMul_1/ReadVariableOp¢gru_cell_5/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿå
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_41670*
condR
while_cond_41669*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¼
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ(
Ð
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_41788

inputs%
forward_gru_1_41598:%
forward_gru_1_41600:%
forward_gru_1_41602:
&
backward_gru_1_41766:&
backward_gru_1_41768:&
backward_gru_1_41770:

identity¢&backward_gru_1/StatefulPartitionedCall¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢%forward_gru_1/StatefulPartitionedCall
%forward_gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsforward_gru_1_41598forward_gru_1_41600forward_gru_1_41602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_41597
&backward_gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_gru_1_41766backward_gru_1_41768backward_gru_1_41770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41765M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ä
concatConcatV2.forward_gru_1/StatefulPartitionedCall:output:0/backward_gru_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpforward_gru_1_41600*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: §
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbackward_gru_1_41768*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp'^backward_gru_1/StatefulPartitionedCallS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp&^forward_gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&backward_gru_1/StatefulPartitionedCall&backward_gru_1/StatefulPartitionedCall2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2N
%forward_gru_1/StatefulPartitionedCall%forward_gru_1/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹<
ø
while_body_46302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_5_readvariableop_resource_0:C
1while_gru_cell_5_matmul_readvariableop_resource_0:E
3while_gru_cell_5_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_5_readvariableop_resource:A
/while_gru_cell_5_matmul_readvariableop_resource:C
1while_gru_cell_5_matmul_1_readvariableop_resource:
¢&while/gru_cell_5/MatMul/ReadVariableOp¢(while/gru_cell_5/MatMul_1/ReadVariableOp¢while/gru_cell_5/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ñ(
Ð
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42201

inputs%
forward_gru_1_42172:%
forward_gru_1_42174:%
forward_gru_1_42176:
&
backward_gru_1_42179:&
backward_gru_1_42181:&
backward_gru_1_42183:

identity¢&backward_gru_1/StatefulPartitionedCall¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢%forward_gru_1/StatefulPartitionedCall
%forward_gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsforward_gru_1_42172forward_gru_1_42174forward_gru_1_42176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_42158
&backward_gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_gru_1_42179backward_gru_1_42181backward_gru_1_42183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_41977M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ä
concatConcatV2.forward_gru_1/StatefulPartitionedCall:output:0/backward_gru_1/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpforward_gru_1_42174*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: §
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbackward_gru_1_42181*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp'^backward_gru_1/StatefulPartitionedCallS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp&^forward_gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&backward_gru_1/StatefulPartitionedCall&backward_gru_1/StatefulPartitionedCall2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2N
%forward_gru_1/StatefulPartitionedCall%forward_gru_1/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_42062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_42062___redundant_placeholder03
/while_while_cond_42062___redundant_placeholder13
/while_while_cond_42062___redundant_placeholder23
/while_while_cond_42062___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
Õ
¥
while_cond_46462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_46462___redundant_placeholder03
/while_while_cond_46462___redundant_placeholder13
/while_while_cond_46462___redundant_placeholder23
/while_while_cond_46462___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
¦
»
-__inference_forward_gru_1_layer_call_fn_45378

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_41597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_41881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_41881___redundant_placeholder03
/while_while_cond_41881___redundant_placeholder13
/while_while_cond_41881___redundant_placeholder23
/while_while_cond_41881___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
N

forward_gru_1_while_body_427278
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_27
3forward_gru_1_while_forward_gru_1_strided_slice_1_0s
oforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0J
8forward_gru_1_while_gru_cell_4_readvariableop_resource_0:Q
?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:S
Aforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
 
forward_gru_1_while_identity"
forward_gru_1_while_identity_1"
forward_gru_1_while_identity_2"
forward_gru_1_while_identity_3"
forward_gru_1_while_identity_45
1forward_gru_1_while_forward_gru_1_strided_slice_1q
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensorH
6forward_gru_1_while_gru_cell_4_readvariableop_resource:O
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:Q
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢-forward_gru_1/while/gru_cell_4/ReadVariableOp
Eforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
7forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemoforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0forward_gru_1_while_placeholderNforward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¦
-forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOp8forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0
&forward_gru_1/while/gru_cell_4/unstackUnpack5forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num´
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ß
%forward_gru_1/while/gru_cell_4/MatMulMatMul>forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0<forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&forward_gru_1/while/gru_cell_4/BiasAddBiasAdd/forward_gru_1/while/gru_cell_4/MatMul:product:0/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$forward_gru_1/while/gru_cell_4/splitSplit7forward_gru_1/while/gru_cell_4/split/split_dim:output:0/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¸
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Æ
'forward_gru_1/while/gru_cell_4/MatMul_1MatMul!forward_gru_1_while_placeholder_2>forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAdd1forward_gru_1/while/gru_cell_4/MatMul_1:product:0/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ{
0forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&forward_gru_1/while/gru_cell_4/split_1SplitV1forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0-forward_gru_1/while/gru_cell_4/Const:output:09forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split½
"forward_gru_1/while/gru_cell_4/addAddV2-forward_gru_1/while/gru_cell_4/split:output:0/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru_1/while/gru_cell_4/SigmoidSigmoid&forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
$forward_gru_1/while/gru_cell_4/add_1AddV2-forward_gru_1/while/gru_cell_4/split:output:1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid(forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
"forward_gru_1/while/gru_cell_4/mulMul,forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
$forward_gru_1/while/gru_cell_4/add_2AddV2-forward_gru_1/while/gru_cell_4/split:output:2&forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#forward_gru_1/while/gru_cell_4/ReluRelu(forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
$forward_gru_1/while/gru_cell_4/mul_1Mul*forward_gru_1/while/gru_cell_4/Sigmoid:y:0!forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
$forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"forward_gru_1/while/gru_cell_4/subSub-forward_gru_1/while/gru_cell_4/sub/x:output:0*forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
$forward_gru_1/while/gru_cell_4/mul_2Mul&forward_gru_1/while/gru_cell_4/sub:z:01forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
$forward_gru_1/while/gru_cell_4/add_3AddV2(forward_gru_1/while/gru_cell_4/mul_1:z:0(forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
8forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!forward_gru_1_while_placeholder_1forward_gru_1_while_placeholder(forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ[
forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/addAddV2forward_gru_1_while_placeholder"forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: ]
forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru_1/while/add_1AddV24forward_gru_1_while_forward_gru_1_while_loop_counter$forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
forward_gru_1/while/IdentityIdentityforward_gru_1/while/add_1:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: ¢
forward_gru_1/while/Identity_1Identity:forward_gru_1_while_forward_gru_1_while_maximum_iterations^forward_gru_1/while/NoOp*
T0*
_output_shapes
: 
forward_gru_1/while/Identity_2Identityforward_gru_1/while/add:z:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: Ã
forward_gru_1/while/Identity_3IdentityHforward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒ¡
forward_gru_1/while/Identity_4Identity(forward_gru_1/while/gru_cell_4/add_3:z:0^forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
forward_gru_1/while/NoOpNoOp5^forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp7^forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp.^forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1forward_gru_1_while_forward_gru_1_strided_slice_13forward_gru_1_while_forward_gru_1_strided_slice_1_0"
?forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceAforward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0"
=forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource?forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"r
6forward_gru_1_while_gru_cell_4_readvariableop_resource8forward_gru_1_while_gru_cell_4_readvariableop_resource_0"E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0"I
forward_gru_1_while_identity_1'forward_gru_1/while/Identity_1:output:0"I
forward_gru_1_while_identity_2'forward_gru_1/while/Identity_2:output:0"I
forward_gru_1_while_identity_3'forward_gru_1/while/Identity_3:output:0"I
forward_gru_1_while_identity_4'forward_gru_1/while/Identity_4:output:0"à
mforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensoroforward_gru_1_while_tensorarrayv2read_tensorlistgetitem_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2l
4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp4forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2p
6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp6forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2^
-forward_gru_1/while/gru_cell_4/ReadVariableOp-forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 


ó
B__inference_dense_2_layer_call_and_return_conditional_losses_45319

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
b

.bidirectional_1_forward_gru_1_while_body_43652X
Tbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_loop_counter^
Zbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_maximum_iterations3
/bidirectional_1_forward_gru_1_while_placeholder5
1bidirectional_1_forward_gru_1_while_placeholder_15
1bidirectional_1_forward_gru_1_while_placeholder_2W
Sbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1_0
bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0Z
Hbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0:a
Obidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0:c
Qbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0:
0
,bidirectional_1_forward_gru_1_while_identity2
.bidirectional_1_forward_gru_1_while_identity_12
.bidirectional_1_forward_gru_1_while_identity_22
.bidirectional_1_forward_gru_1_while_identity_32
.bidirectional_1_forward_gru_1_while_identity_4U
Qbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1
bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensorX
Fbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource:_
Mbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource:a
Obidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource:
¢Dbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp¢Fbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp¢=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp¦
Ubidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ½
Gbidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0/bidirectional_1_forward_gru_1_while_placeholder^bidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Æ
=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOpReadVariableOpHbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0*
_output_shapes

:*
dtype0½
6bidirectional_1/forward_gru_1/while/gru_cell_4/unstackUnpackEbidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÔ
Dbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOpObidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
5bidirectional_1/forward_gru_1/while/gru_cell_4/MatMulMatMulNbidirectional_1/forward_gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0Lbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
6bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAddBiasAdd?bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul:product:0?bidirectional_1/forward_gru_1/while/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>bidirectional_1/forward_gru_1/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ°
4bidirectional_1/forward_gru_1/while/gru_cell_4/splitSplitGbidirectional_1/forward_gru_1/while/gru_cell_4/split/split_dim:output:0?bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitØ
Fbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpQbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0ö
7bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1MatMul1bidirectional_1_forward_gru_1_while_placeholder_2Nbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
8bidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd_1BiasAddAbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1:product:0?bidirectional_1/forward_gru_1/while/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4bidirectional_1/forward_gru_1/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
@bidirectional_1/forward_gru_1/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
6bidirectional_1/forward_gru_1/while/gru_cell_4/split_1SplitVAbidirectional_1/forward_gru_1/while/gru_cell_4/BiasAdd_1:output:0=bidirectional_1/forward_gru_1/while/gru_cell_4/Const:output:0Ibidirectional_1/forward_gru_1/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splití
2bidirectional_1/forward_gru_1/while/gru_cell_4/addAddV2=bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:0?bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
6bidirectional_1/forward_gru_1/while/gru_cell_4/SigmoidSigmoid6bidirectional_1/forward_gru_1/while/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ï
4bidirectional_1/forward_gru_1/while/gru_cell_4/add_1AddV2=bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:1?bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
8bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid_1Sigmoid8bidirectional_1/forward_gru_1/while/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ê
2bidirectional_1/forward_gru_1/while/gru_cell_4/mulMul<bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid_1:y:0?bidirectional_1/forward_gru_1/while/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
4bidirectional_1/forward_gru_1/while/gru_cell_4/add_2AddV2=bidirectional_1/forward_gru_1/while/gru_cell_4/split:output:26bidirectional_1/forward_gru_1/while/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
3bidirectional_1/forward_gru_1/while/gru_cell_4/ReluRelu8bidirectional_1/forward_gru_1/while/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ü
4bidirectional_1/forward_gru_1/while/gru_cell_4/mul_1Mul:bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid:y:01bidirectional_1_forward_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
4bidirectional_1/forward_gru_1/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?æ
2bidirectional_1/forward_gru_1/while/gru_cell_4/subSub=bidirectional_1/forward_gru_1/while/gru_cell_4/sub/x:output:0:bidirectional_1/forward_gru_1/while/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
è
4bidirectional_1/forward_gru_1/while/gru_cell_4/mul_2Mul6bidirectional_1/forward_gru_1/while/gru_cell_4/sub:z:0Abidirectional_1/forward_gru_1/while/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ã
4bidirectional_1/forward_gru_1/while/gru_cell_4/add_3AddV28bidirectional_1/forward_gru_1/while/gru_cell_4/mul_1:z:08bidirectional_1/forward_gru_1/while/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
Hbidirectional_1/forward_gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem1bidirectional_1_forward_gru_1_while_placeholder_1/bidirectional_1_forward_gru_1_while_placeholder8bidirectional_1/forward_gru_1/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒk
)bidirectional_1/forward_gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¶
'bidirectional_1/forward_gru_1/while/addAddV2/bidirectional_1_forward_gru_1_while_placeholder2bidirectional_1/forward_gru_1/while/add/y:output:0*
T0*
_output_shapes
: m
+bidirectional_1/forward_gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ß
)bidirectional_1/forward_gru_1/while/add_1AddV2Tbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_loop_counter4bidirectional_1/forward_gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: ³
,bidirectional_1/forward_gru_1/while/IdentityIdentity-bidirectional_1/forward_gru_1/while/add_1:z:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: â
.bidirectional_1/forward_gru_1/while/Identity_1IdentityZbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_while_maximum_iterations)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: ³
.bidirectional_1/forward_gru_1/while/Identity_2Identity+bidirectional_1/forward_gru_1/while/add:z:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: ó
.bidirectional_1/forward_gru_1/while/Identity_3IdentityXbidirectional_1/forward_gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*
_output_shapes
: :éèÒÑ
.bidirectional_1/forward_gru_1/while/Identity_4Identity8bidirectional_1/forward_gru_1/while/gru_cell_4/add_3:z:0)^bidirectional_1/forward_gru_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
(bidirectional_1/forward_gru_1/while/NoOpNoOpE^bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpG^bidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp>^bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "¨
Qbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1Sbidirectional_1_forward_gru_1_while_bidirectional_1_forward_gru_1_strided_slice_1_0"¤
Obidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resourceQbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_1_readvariableop_resource_0" 
Mbidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resourceObidirectional_1_forward_gru_1_while_gru_cell_4_matmul_readvariableop_resource_0"
Fbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resourceHbidirectional_1_forward_gru_1_while_gru_cell_4_readvariableop_resource_0"e
,bidirectional_1_forward_gru_1_while_identity5bidirectional_1/forward_gru_1/while/Identity:output:0"i
.bidirectional_1_forward_gru_1_while_identity_17bidirectional_1/forward_gru_1/while/Identity_1:output:0"i
.bidirectional_1_forward_gru_1_while_identity_27bidirectional_1/forward_gru_1/while/Identity_2:output:0"i
.bidirectional_1_forward_gru_1_while_identity_37bidirectional_1/forward_gru_1/while/Identity_3:output:0"i
.bidirectional_1_forward_gru_1_while_identity_47bidirectional_1/forward_gru_1/while/Identity_4:output:0"¢
bidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensorbidirectional_1_forward_gru_1_while_tensorarrayv2read_tensorlistgetitem_bidirectional_1_forward_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Dbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOpDbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul/ReadVariableOp2
Fbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOpFbidirectional_1/forward_gru_1/while/gru_cell_4/MatMul_1/ReadVariableOp2~
=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp=bidirectional_1/forward_gru_1/while/gru_cell_4/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ß
¡
while_body_40987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_4_41009_0:*
while_gru_cell_4_41011_0:*
while_gru_cell_4_41013_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_4_41009:(
while_gru_cell_4_41011:(
while_gru_cell_4_41013:
¢(while/gru_cell_4/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_41009_0while_gru_cell_4_41011_0while_gru_cell_4_41013_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_40935Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_4_41009while_gru_cell_4_41009_0"2
while_gru_cell_4_41011while_gru_cell_4_41011_0"2
while_gru_cell_4_41013while_gru_cell_4_41013_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
¾

'__inference_dense_3_layer_call_fn_45328

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¡
while_body_41157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_5_41179_0:*
while_gru_cell_5_41181_0:*
while_gru_cell_5_41183_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_5_41179:(
while_gru_cell_5_41181:(
while_gru_cell_5_41183:
¢(while/gru_cell_5/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_5_41179_0while_gru_cell_5_41181_0while_gru_cell_5_41183_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_41144Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity1while/gru_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_5_41179while_gru_cell_5_41179_0"2
while_gru_cell_5_41181while_gru_cell_5_41181_0"2
while_gru_cell_5_41183while_gru_cell_5_41183_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2T
(while/gru_cell_5/StatefulPartitionedCall(while/gru_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 


ó
B__inference_dense_3_layer_call_and_return_conditional_losses_42582

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â<
ø
while_body_41670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_5_readvariableop_resource_0:C
1while_gru_cell_5_matmul_readvariableop_resource_0:E
3while_gru_cell_5_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_5_readvariableop_resource:A
/while_gru_cell_5_matmul_readvariableop_resource:C
1while_gru_cell_5_matmul_1_readvariableop_resource:
¢&while/gru_cell_5/MatMul/ReadVariableOp¢(while/gru_cell_5/MatMul_1/ReadVariableOp¢while/gru_cell_5/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
è
Þ	
;sequential_1_bidirectional_1_forward_gru_1_while_cond_40447r
nsequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_loop_counterx
tsequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_maximum_iterations@
<sequential_1_bidirectional_1_forward_gru_1_while_placeholderB
>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_1B
>sequential_1_bidirectional_1_forward_gru_1_while_placeholder_2t
psequential_1_bidirectional_1_forward_gru_1_while_less_sequential_1_bidirectional_1_forward_gru_1_strided_slice_1
sequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_cond_40447___redundant_placeholder0
sequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_cond_40447___redundant_placeholder1
sequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_cond_40447___redundant_placeholder2
sequential_1_bidirectional_1_forward_gru_1_while_sequential_1_bidirectional_1_forward_gru_1_while_cond_40447___redundant_placeholder3=
9sequential_1_bidirectional_1_forward_gru_1_while_identity

5sequential_1/bidirectional_1/forward_gru_1/while/LessLess<sequential_1_bidirectional_1_forward_gru_1_while_placeholderpsequential_1_bidirectional_1_forward_gru_1_while_less_sequential_1_bidirectional_1_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: ¡
9sequential_1/bidirectional_1/forward_gru_1/while/IdentityIdentity9sequential_1/bidirectional_1/forward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "
9sequential_1_bidirectional_1_forward_gru_1_while_identityBsequential_1/bidirectional_1/forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
Y
Ù
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45866

inputs4
"gru_cell_4_readvariableop_resource:;
)gru_cell_4_matmul_readvariableop_resource:=
+gru_cell_4_matmul_1_readvariableop_resource:

identity¢Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_4/MatMul/ReadVariableOp¢"gru_cell_4/MatMul_1/ReadVariableOp¢gru_cell_4/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿà
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask|
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_45771*
condR
while_cond_45770*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    »
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ð
Bbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SquareSquareYbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       û
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/SumSumFbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square:y:0Jbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Abidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ý
?bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mulMulJbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/mul/x:output:0Hbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpR^bidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2¦
Qbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOpQbidirectional_1/forward_gru_1/gru_cell_4/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
«
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_46978

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ò
Cbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SquareSquareZbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       þ
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/SumSumGbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square:y:0Kbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
Bbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8
@bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mulMulKbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/mul/x:output:0Ibidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpS^bidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2¨
Rbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOpRbidirectional_1/backward_gru_1/gru_cell_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
states/0
 
·
 __inference__wrapped_model_40704
bidirectional_1_input_
Msequential_1_bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resource:f
Tsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource:h
Vsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource:
`
Nsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resource:g
Usequential_1_bidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource:i
Wsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource:
E
3sequential_1_dense_2_matmul_readvariableop_resource:B
4sequential_1_dense_2_biasadd_readvariableop_resource:E
3sequential_1_dense_3_matmul_readvariableop_resource:B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity¢Lsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp¢Nsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp¢Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp¢1sequential_1/bidirectional_1/backward_gru_1/while¢Ksequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp¢Msequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp¢Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp¢0sequential_1/bidirectional_1/forward_gru_1/while¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢*sequential_1/dense_2/MatMul/ReadVariableOp¢+sequential_1/dense_3/BiasAdd/ReadVariableOp¢*sequential_1/dense_3/MatMul/ReadVariableOpu
0sequential_1/bidirectional_1/forward_gru_1/ShapeShapebidirectional_1_input*
T0*
_output_shapes
:
>sequential_1/bidirectional_1/forward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
@sequential_1/bidirectional_1/forward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@sequential_1/bidirectional_1/forward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¨
8sequential_1/bidirectional_1/forward_gru_1/strided_sliceStridedSlice9sequential_1/bidirectional_1/forward_gru_1/Shape:output:0Gsequential_1/bidirectional_1/forward_gru_1/strided_slice/stack:output:0Isequential_1/bidirectional_1/forward_gru_1/strided_slice/stack_1:output:0Isequential_1/bidirectional_1/forward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9sequential_1/bidirectional_1/forward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
ô
7sequential_1/bidirectional_1/forward_gru_1/zeros/packedPackAsequential_1/bidirectional_1/forward_gru_1/strided_slice:output:0Bsequential_1/bidirectional_1/forward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:{
6sequential_1/bidirectional_1/forward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    í
0sequential_1/bidirectional_1/forward_gru_1/zerosFill@sequential_1/bidirectional_1/forward_gru_1/zeros/packed:output:0?sequential_1/bidirectional_1/forward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

9sequential_1/bidirectional_1/forward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
4sequential_1/bidirectional_1/forward_gru_1/transpose	Transposebidirectional_1_inputBsequential_1/bidirectional_1/forward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2sequential_1/bidirectional_1/forward_gru_1/Shape_1Shape8sequential_1/bidirectional_1/forward_gru_1/transpose:y:0*
T0*
_output_shapes
:
@sequential_1/bidirectional_1/forward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Bsequential_1/bidirectional_1/forward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Bsequential_1/bidirectional_1/forward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
:sequential_1/bidirectional_1/forward_gru_1/strided_slice_1StridedSlice;sequential_1/bidirectional_1/forward_gru_1/Shape_1:output:0Isequential_1/bidirectional_1/forward_gru_1/strided_slice_1/stack:output:0Ksequential_1/bidirectional_1/forward_gru_1/strided_slice_1/stack_1:output:0Ksequential_1/bidirectional_1/forward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Fsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿµ
8sequential_1/bidirectional_1/forward_gru_1/TensorArrayV2TensorListReserveOsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2/element_shape:output:0Csequential_1/bidirectional_1/forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ±
`sequential_1/bidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   á
Rsequential_1/bidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor8sequential_1/bidirectional_1/forward_gru_1/transpose:y:0isequential_1/bidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
@sequential_1/bidirectional_1/forward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Bsequential_1/bidirectional_1/forward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Bsequential_1/bidirectional_1/forward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
:sequential_1/bidirectional_1/forward_gru_1/strided_slice_2StridedSlice8sequential_1/bidirectional_1/forward_gru_1/transpose:y:0Isequential_1/bidirectional_1/forward_gru_1/strided_slice_2/stack:output:0Ksequential_1/bidirectional_1/forward_gru_1/strided_slice_2/stack_1:output:0Ksequential_1/bidirectional_1/forward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÒ
Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOpReadVariableOpMsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resource*
_output_shapes

:*
dtype0Ë
=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/unstackUnpackLsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numà
Ksequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpReadVariableOpTsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
<sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMulMatMulCsequential_1/bidirectional_1/forward_gru_1/strided_slice_2:output:0Ssequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/BiasAddBiasAddFsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul:product:0Fsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/splitSplitNsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split/split_dim:output:0Fsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitä
Msequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpVsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
>sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1MatMul9sequential_1/bidirectional_1/forward_gru_1/zeros:output:0Usequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd_1BiasAddHsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1:product:0Fsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Gsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split_1SplitVHsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/BiasAdd_1:output:0Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/Const:output:0Psequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
9sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/addAddV2Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split:output:0Fsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/SigmoidSigmoid=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/add_1AddV2Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split:output:1Fsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
?sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid_1Sigmoid?sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/mulMulCsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid_1:y:0Fsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/add_2AddV2Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/split:output:2=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
:sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReluRelu?sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/mul_1MulAsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid:y:09sequential_1/bidirectional_1/forward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?û
9sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/subSubDsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/sub/x:output:0Asequential_1/bidirectional_1/forward_gru_1/gru_cell_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ý
;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/mul_2Mul=sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/sub:z:0Hsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ø
;sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/add_3AddV2?sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/mul_1:z:0?sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Hsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¹
:sequential_1/bidirectional_1/forward_gru_1/TensorArrayV2_1TensorListReserveQsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2_1/element_shape:output:0Csequential_1/bidirectional_1/forward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒq
/sequential_1/bidirectional_1/forward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Csequential_1/bidirectional_1/forward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
=sequential_1/bidirectional_1/forward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è	
0sequential_1/bidirectional_1/forward_gru_1/whileWhileFsequential_1/bidirectional_1/forward_gru_1/while/loop_counter:output:0Lsequential_1/bidirectional_1/forward_gru_1/while/maximum_iterations:output:08sequential_1/bidirectional_1/forward_gru_1/time:output:0Csequential_1/bidirectional_1/forward_gru_1/TensorArrayV2_1:handle:09sequential_1/bidirectional_1/forward_gru_1/zeros:output:0Csequential_1/bidirectional_1/forward_gru_1/strided_slice_1:output:0bsequential_1/bidirectional_1/forward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_1_bidirectional_1_forward_gru_1_gru_cell_4_readvariableop_resourceTsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_matmul_readvariableop_resourceVsequential_1_bidirectional_1_forward_gru_1_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *G
body?R=
;sequential_1_bidirectional_1_forward_gru_1_while_body_40448*G
cond?R=
;sequential_1_bidirectional_1_forward_gru_1_while_cond_40447*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations ¬
[sequential_1/bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ã
Msequential_1/bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStack9sequential_1/bidirectional_1/forward_gru_1/while:output:3dsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
@sequential_1/bidirectional_1/forward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Bsequential_1/bidirectional_1/forward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Bsequential_1/bidirectional_1/forward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
:sequential_1/bidirectional_1/forward_gru_1/strided_slice_3StridedSliceVsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0Isequential_1/bidirectional_1/forward_gru_1/strided_slice_3/stack:output:0Ksequential_1/bidirectional_1/forward_gru_1/strided_slice_3/stack_1:output:0Ksequential_1/bidirectional_1/forward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
;sequential_1/bidirectional_1/forward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
6sequential_1/bidirectional_1/forward_gru_1/transpose_1	TransposeVsequential_1/bidirectional_1/forward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0Dsequential_1/bidirectional_1/forward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2sequential_1/bidirectional_1/forward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    v
1sequential_1/bidirectional_1/backward_gru_1/ShapeShapebidirectional_1_input*
T0*
_output_shapes
:
?sequential_1/bidirectional_1/backward_gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Asequential_1/bidirectional_1/backward_gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Asequential_1/bidirectional_1/backward_gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9sequential_1/bidirectional_1/backward_gru_1/strided_sliceStridedSlice:sequential_1/bidirectional_1/backward_gru_1/Shape:output:0Hsequential_1/bidirectional_1/backward_gru_1/strided_slice/stack:output:0Jsequential_1/bidirectional_1/backward_gru_1/strided_slice/stack_1:output:0Jsequential_1/bidirectional_1/backward_gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential_1/bidirectional_1/backward_gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
÷
8sequential_1/bidirectional_1/backward_gru_1/zeros/packedPackBsequential_1/bidirectional_1/backward_gru_1/strided_slice:output:0Csequential_1/bidirectional_1/backward_gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:|
7sequential_1/bidirectional_1/backward_gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ð
1sequential_1/bidirectional_1/backward_gru_1/zerosFillAsequential_1/bidirectional_1/backward_gru_1/zeros/packed:output:0@sequential_1/bidirectional_1/backward_gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

:sequential_1/bidirectional_1/backward_gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ô
5sequential_1/bidirectional_1/backward_gru_1/transpose	Transposebidirectional_1_inputCsequential_1/bidirectional_1/backward_gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3sequential_1/bidirectional_1/backward_gru_1/Shape_1Shape9sequential_1/bidirectional_1/backward_gru_1/transpose:y:0*
T0*
_output_shapes
:
Asequential_1/bidirectional_1/backward_gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Csequential_1/bidirectional_1/backward_gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Csequential_1/bidirectional_1/backward_gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
;sequential_1/bidirectional_1/backward_gru_1/strided_slice_1StridedSlice<sequential_1/bidirectional_1/backward_gru_1/Shape_1:output:0Jsequential_1/bidirectional_1/backward_gru_1/strided_slice_1/stack:output:0Lsequential_1/bidirectional_1/backward_gru_1/strided_slice_1/stack_1:output:0Lsequential_1/bidirectional_1/backward_gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Gsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¸
9sequential_1/bidirectional_1/backward_gru_1/TensorArrayV2TensorListReservePsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2/element_shape:output:0Dsequential_1/bidirectional_1/backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
:sequential_1/bidirectional_1/backward_gru_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ø
5sequential_1/bidirectional_1/backward_gru_1/ReverseV2	ReverseV29sequential_1/bidirectional_1/backward_gru_1/transpose:y:0Csequential_1/bidirectional_1/backward_gru_1/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
asequential_1/bidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   é
Ssequential_1/bidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor>sequential_1/bidirectional_1/backward_gru_1/ReverseV2:output:0jsequential_1/bidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Asequential_1/bidirectional_1/backward_gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Csequential_1/bidirectional_1/backward_gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Csequential_1/bidirectional_1/backward_gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
;sequential_1/bidirectional_1/backward_gru_1/strided_slice_2StridedSlice9sequential_1/bidirectional_1/backward_gru_1/transpose:y:0Jsequential_1/bidirectional_1/backward_gru_1/strided_slice_2/stack:output:0Lsequential_1/bidirectional_1/backward_gru_1/strided_slice_2/stack_1:output:0Lsequential_1/bidirectional_1/backward_gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÔ
Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOpReadVariableOpNsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes

:*
dtype0Í
>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/unstackUnpackMsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numâ
Lsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOpUsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
=sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMulMatMulDsequential_1/bidirectional_1/backward_gru_1/strided_slice_2:output:0Tsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/BiasAddBiasAddGsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul:product:0Gsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/splitSplitOsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split/split_dim:output:0Gsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitæ
Nsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpWsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
?sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1MatMul:sequential_1/bidirectional_1/backward_gru_1/zeros:output:0Vsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd_1BiasAddIsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1:product:0Gsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Hsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split_1SplitVIsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/BiasAdd_1:output:0Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/Const:output:0Qsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
:sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/addAddV2Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split:output:0Gsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/SigmoidSigmoid>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/add_1AddV2Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split:output:1Gsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
@sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid_1Sigmoid@sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

:sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/mulMulDsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid_1:y:0Gsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/add_2AddV2Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/split:output:2>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
;sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReluRelu@sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
õ
<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/mul_1MulBsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid:y:0:sequential_1/bidirectional_1/backward_gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?þ
:sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/subSubEsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/sub/x:output:0Bsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/mul_2Mul>sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/sub:z:0Isequential_1/bidirectional_1/backward_gru_1/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
<sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/add_3AddV2@sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/mul_1:z:0@sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Isequential_1/bidirectional_1/backward_gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ¼
;sequential_1/bidirectional_1/backward_gru_1/TensorArrayV2_1TensorListReserveRsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2_1/element_shape:output:0Dsequential_1/bidirectional_1/backward_gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
0sequential_1/bidirectional_1/backward_gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 
Dsequential_1/bidirectional_1/backward_gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
>sequential_1/bidirectional_1/backward_gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : õ	
1sequential_1/bidirectional_1/backward_gru_1/whileWhileGsequential_1/bidirectional_1/backward_gru_1/while/loop_counter:output:0Msequential_1/bidirectional_1/backward_gru_1/while/maximum_iterations:output:09sequential_1/bidirectional_1/backward_gru_1/time:output:0Dsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2_1:handle:0:sequential_1/bidirectional_1/backward_gru_1/zeros:output:0Dsequential_1/bidirectional_1/backward_gru_1/strided_slice_1:output:0csequential_1/bidirectional_1/backward_gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Nsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_readvariableop_resourceUsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_matmul_readvariableop_resourceWsequential_1_bidirectional_1_backward_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *H
body@R>
<sequential_1_bidirectional_1_backward_gru_1_while_body_40599*H
cond@R>
<sequential_1_bidirectional_1_backward_gru_1_while_cond_40598*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations ­
\sequential_1/bidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Æ
Nsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStackTensorListStack:sequential_1/bidirectional_1/backward_gru_1/while:output:3esequential_1/bidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
Asequential_1/bidirectional_1/backward_gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
Csequential_1/bidirectional_1/backward_gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Csequential_1/bidirectional_1/backward_gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
;sequential_1/bidirectional_1/backward_gru_1/strided_slice_3StridedSliceWsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0Jsequential_1/bidirectional_1/backward_gru_1/strided_slice_3/stack:output:0Lsequential_1/bidirectional_1/backward_gru_1/strided_slice_3/stack_1:output:0Lsequential_1/bidirectional_1/backward_gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
<sequential_1/bidirectional_1/backward_gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
7sequential_1/bidirectional_1/backward_gru_1/transpose_1	TransposeWsequential_1/bidirectional_1/backward_gru_1/TensorArrayV2Stack/TensorListStack:tensor:0Esequential_1/bidirectional_1/backward_gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

3sequential_1/bidirectional_1/backward_gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    j
(sequential_1/bidirectional_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¨
#sequential_1/bidirectional_1/concatConcatV2Csequential_1/bidirectional_1/forward_gru_1/strided_slice_3:output:0Dsequential_1/bidirectional_1/backward_gru_1/strided_slice_3:output:01sequential_1/bidirectional_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¹
sequential_1/dense_2/MatMulMatMul,sequential_1/bidirectional_1/concat:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0´
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOpM^sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpO^sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpF^sequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp2^sequential_1/bidirectional_1/backward_gru_1/whileL^sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpN^sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpE^sequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp1^sequential_1/bidirectional_1/forward_gru_1/while,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2
Lsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOpLsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul/ReadVariableOp2 
Nsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOpNsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/MatMul_1/ReadVariableOp2
Esequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOpEsequential_1/bidirectional_1/backward_gru_1/gru_cell_5/ReadVariableOp2f
1sequential_1/bidirectional_1/backward_gru_1/while1sequential_1/bidirectional_1/backward_gru_1/while2
Ksequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOpKsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul/ReadVariableOp2
Msequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOpMsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/MatMul_1/ReadVariableOp2
Dsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOpDsequential_1/bidirectional_1/forward_gru_1/gru_cell_4/ReadVariableOp2d
0sequential_1/bidirectional_1/forward_gru_1/while0sequential_1/bidirectional_1/forward_gru_1/while2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:b ^
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namebidirectional_1_input
Ð

¯
forward_gru_1_while_cond_447268
4forward_gru_1_while_forward_gru_1_while_loop_counter>
:forward_gru_1_while_forward_gru_1_while_maximum_iterations#
forward_gru_1_while_placeholder%
!forward_gru_1_while_placeholder_1%
!forward_gru_1_while_placeholder_2:
6forward_gru_1_while_less_forward_gru_1_strided_slice_1O
Kforward_gru_1_while_forward_gru_1_while_cond_44726___redundant_placeholder0O
Kforward_gru_1_while_forward_gru_1_while_cond_44726___redundant_placeholder1O
Kforward_gru_1_while_forward_gru_1_while_cond_44726___redundant_placeholder2O
Kforward_gru_1_while_forward_gru_1_while_cond_44726___redundant_placeholder3 
forward_gru_1_while_identity

forward_gru_1/while/LessLessforward_gru_1_while_placeholder6forward_gru_1_while_less_forward_gru_1_strided_slice_1*
T0*
_output_shapes
: g
forward_gru_1/while/IdentityIdentityforward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "E
forward_gru_1_while_identity%forward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:
´	

/__inference_bidirectional_1_layer_call_fn_43993
inputs_0
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_42201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Â<
ø
while_body_41882
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_5_readvariableop_resource_0:C
1while_gru_cell_5_matmul_readvariableop_resource_0:E
3while_gru_cell_5_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_5_readvariableop_resource:A
/while_gru_cell_5_matmul_readvariableop_resource:C
1while_gru_cell_5_matmul_1_readvariableop_resource:
¢&while/gru_cell_5/MatMul/ReadVariableOp¢(while/gru_cell_5/MatMul_1/ReadVariableOp¢while/gru_cell_5/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒw
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
: 
ë

Â
backward_gru_1_while_cond_44241:
6backward_gru_1_while_backward_gru_1_while_loop_counter@
<backward_gru_1_while_backward_gru_1_while_maximum_iterations$
 backward_gru_1_while_placeholder&
"backward_gru_1_while_placeholder_1&
"backward_gru_1_while_placeholder_2<
8backward_gru_1_while_less_backward_gru_1_strided_slice_1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44241___redundant_placeholder0Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44241___redundant_placeholder1Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44241___redundant_placeholder2Q
Mbackward_gru_1_while_backward_gru_1_while_cond_44241___redundant_placeholder3!
backward_gru_1_while_identity

backward_gru_1/while/LessLess backward_gru_1_while_placeholder8backward_gru_1_while_less_backward_gru_1_strided_slice_1*
T0*
_output_shapes
: i
backward_gru_1/while/IdentityIdentitybackward_gru_1/while/Less:z:0*
T0
*
_output_shapes
: "G
backward_gru_1_while_identity&backward_gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ
: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:

_output_shapes
: :

_output_shapes
:"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ê
serving_default¶
[
bidirectional_1_inputB
'serving_default_bidirectional_1_input:0ÿÿÿÿÿÿÿÿÿ;
dense_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:àÇ
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ì
forward_layer
backward_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer

%iter

&beta_1

'beta_2
	(decay
)learning_ratemmmm*m+m,m-m.m/mvvvv*v+v,v-v.v/v"
	optimizer
f
*0
+1
,2
-3
.4
/5
6
7
8
9"
trackable_list_wrapper
f
*0
+1
,2
-3
.4
/5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_1_layer_call_fn_42624
,__inference_sequential_1_layer_call_fn_43231
,__inference_sequential_1_layer_call_fn_43256
,__inference_sequential_1_layer_call_fn_43110À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_43588
G__inference_sequential_1_layer_call_and_return_conditional_losses_43920
G__inference_sequential_1_layer_call_and_return_conditional_losses_43149
G__inference_sequential_1_layer_call_and_return_conditional_losses_43188À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÙBÖ
 __inference__wrapped_model_40704bidirectional_1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
5serving_default"
signature_map
Ú
6cell
7
state_spec
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<_random_generator
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Ú
?cell
@
state_spec
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E_random_generator
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
J
*0
+1
,2
-3
.4
/5"
trackable_list_wrapper
J
*0
+1
,2
-3
.4
/5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
°2­
/__inference_bidirectional_1_layer_call_fn_43976
/__inference_bidirectional_1_layer_call_fn_43993
/__inference_bidirectional_1_layer_call_fn_44010
/__inference_bidirectional_1_layer_call_fn_44027æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44345
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44663
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44981
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_45299æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :2dense_2/kernel
:2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_2_layer_call_fn_45308¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_2_layer_call_and_return_conditional_losses_45319¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_3/kernel
:2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_3_layer_call_fn_45328¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_3_layer_call_and_return_conditional_losses_45339¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
A:?2/bidirectional_1/forward_gru_1/gru_cell_4/kernel
K:I
29bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel
?:=2-bidirectional_1/forward_gru_1/gru_cell_4/bias
B:@20bidirectional_1/backward_gru_1/gru_cell_5/kernel
L:J
2:bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel
@:>2.bidirectional_1/backward_gru_1/gru_cell_5/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ØBÕ
#__inference_signature_wrapper_43947bidirectional_1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è

*kernel
+recurrent_kernel
,bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]_random_generator
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
¹

astates
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
8	variables
9trainable_variables
:regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
-__inference_forward_gru_1_layer_call_fn_45356
-__inference_forward_gru_1_layer_call_fn_45367
-__inference_forward_gru_1_layer_call_fn_45378
-__inference_forward_gru_1_layer_call_fn_45389Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45548
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45707
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45866
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_46025Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
è

-kernel
.recurrent_kernel
/bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k_random_generator
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
¹

ostates
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
.__inference_backward_gru_1_layer_call_fn_46042
.__inference_backward_gru_1_layer_call_fn_46053
.__inference_backward_gru_1_layer_call_fn_46064
.__inference_backward_gru_1_layer_call_fn_46075Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46236
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46397
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46558
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46719Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	utotal
	vcount
w	variables
x	keras_api"
_tf_keras_metric
^
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api"
_tf_keras_metric
5
*0
+1
,2"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_gru_cell_4_layer_call_fn_46739
*__inference_gru_cell_4_layer_call_fn_46753¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_46798
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_46843¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
__inference_loss_fn_0_46854
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
g	variables
htrainable_variables
iregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
*__inference_gru_cell_5_layer_call_fn_46874
*__inference_gru_cell_5_layer_call_fn_46888¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_46933
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_46978¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
__inference_loss_fn_1_46989
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
u0
v1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
n0"
trackable_list_wrapper
 "
trackable_dict_wrapper
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
F:D26Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/m
P:N
2@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/m
D:B24Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/m
G:E27Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/m
Q:O
2AAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/m
E:C25Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/m
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
F:D26Adam/bidirectional_1/forward_gru_1/gru_cell_4/kernel/v
P:N
2@Adam/bidirectional_1/forward_gru_1/gru_cell_4/recurrent_kernel/v
D:B24Adam/bidirectional_1/forward_gru_1/gru_cell_4/bias/v
G:E27Adam/bidirectional_1/backward_gru_1/gru_cell_5/kernel/v
Q:O
2AAdam/bidirectional_1/backward_gru_1/gru_cell_5/recurrent_kernel/v
E:C25Adam/bidirectional_1/backward_gru_1/gru_cell_5/bias/v¨
 __inference__wrapped_model_40704
,*+/-.B¢?
8¢5
30
bidirectional_1_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿÊ
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46236}/-.O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ê
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46397}/-.O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ì
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46558/-.Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ì
I__inference_backward_gru_1_layer_call_and_return_conditional_losses_46719/-.Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¢
.__inference_backward_gru_1_layer_call_fn_46042p/-.O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¢
.__inference_backward_gru_1_layer_call_fn_46053p/-.O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¤
.__inference_backward_gru_1_layer_call_fn_46064r/-.Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¤
.__inference_backward_gru_1_layer_call_fn_46075r/-.Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ü
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44345,*+/-.\¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44663,*+/-.\¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_44981t,*+/-.C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_bidirectional_1_layer_call_and_return_conditional_losses_45299t,*+/-.C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
/__inference_bidirectional_1_layer_call_fn_43976,*+/-.\¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ´
/__inference_bidirectional_1_layer_call_fn_43993,*+/-.\¢Y
R¢O
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_bidirectional_1_layer_call_fn_44010g,*+/-.C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_bidirectional_1_layer_call_fn_44027g,*+/-.C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_2_layer_call_and_return_conditional_losses_45319\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_2_layer_call_fn_45308O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_3_layer_call_and_return_conditional_losses_45339\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_3_layer_call_fn_45328O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÉ
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45548},*+O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 É
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45707},*+O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ë
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_45866,*+Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ë
H__inference_forward_gru_1_layer_call_and_return_conditional_losses_46025,*+Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¡
-__inference_forward_gru_1_layer_call_fn_45356p,*+O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¡
-__inference_forward_gru_1_layer_call_fn_45367p,*+O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
£
-__inference_forward_gru_1_layer_call_fn_45378r,*+Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
£
-__inference_forward_gru_1_layer_call_fn_45389r,*+Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ

E__inference_gru_cell_4_layer_call_and_return_conditional_losses_46798·,*+\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ

$!

0/1/0ÿÿÿÿÿÿÿÿÿ

 
E__inference_gru_cell_4_layer_call_and_return_conditional_losses_46843·,*+\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ

$!

0/1/0ÿÿÿÿÿÿÿÿÿ

 Ø
*__inference_gru_cell_4_layer_call_fn_46739©,*+\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ

"

1/0ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_gru_cell_4_layer_call_fn_46753©,*+\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ

"

1/0ÿÿÿÿÿÿÿÿÿ

E__inference_gru_cell_5_layer_call_and_return_conditional_losses_46933·/-.\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ

$!

0/1/0ÿÿÿÿÿÿÿÿÿ

 
E__inference_gru_cell_5_layer_call_and_return_conditional_losses_46978·/-.\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ

$!

0/1/0ÿÿÿÿÿÿÿÿÿ

 Ø
*__inference_gru_cell_5_layer_call_fn_46874©/-.\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ

"

1/0ÿÿÿÿÿÿÿÿÿ
Ø
*__inference_gru_cell_5_layer_call_fn_46888©/-.\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ

p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ

"

1/0ÿÿÿÿÿÿÿÿÿ
:
__inference_loss_fn_0_46854*¢

¢ 
ª " :
__inference_loss_fn_1_46989-¢

¢ 
ª " Ê
G__inference_sequential_1_layer_call_and_return_conditional_losses_43149
,*+/-.J¢G
@¢=
30
bidirectional_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
G__inference_sequential_1_layer_call_and_return_conditional_losses_43188
,*+/-.J¢G
@¢=
30
bidirectional_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
G__inference_sequential_1_layer_call_and_return_conditional_losses_43588p
,*+/-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
G__inference_sequential_1_layer_call_and_return_conditional_losses_43920p
,*+/-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
,__inference_sequential_1_layer_call_fn_42624r
,*+/-.J¢G
@¢=
30
bidirectional_1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¢
,__inference_sequential_1_layer_call_fn_43110r
,*+/-.J¢G
@¢=
30
bidirectional_1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_43231c
,*+/-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_43256c
,*+/-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÄ
#__inference_signature_wrapper_43947
,*+/-.[¢X
¢ 
QªN
L
bidirectional_1_input30
bidirectional_1_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ