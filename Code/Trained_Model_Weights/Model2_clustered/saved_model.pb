µÁ;
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
"serve*2.8.22v2.8.2-0-g2ea19cbb5758·Û9
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
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
²
+bidirectional/forward_gru/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+bidirectional/forward_gru/gru_cell_1/kernel
«
?bidirectional/forward_gru/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOp+bidirectional/forward_gru/gru_cell_1/kernel*
_output_shapes

:*
dtype0
Æ
5bidirectional/forward_gru/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*F
shared_name75bidirectional/forward_gru/gru_cell_1/recurrent_kernel
¿
Ibidirectional/forward_gru/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp5bidirectional/forward_gru/gru_cell_1/recurrent_kernel*
_output_shapes

:
*
dtype0
®
)bidirectional/forward_gru/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)bidirectional/forward_gru/gru_cell_1/bias
§
=bidirectional/forward_gru/gru_cell_1/bias/Read/ReadVariableOpReadVariableOp)bidirectional/forward_gru/gru_cell_1/bias*
_output_shapes

:*
dtype0
´
,bidirectional/backward_gru/gru_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,bidirectional/backward_gru/gru_cell_2/kernel
­
@bidirectional/backward_gru/gru_cell_2/kernel/Read/ReadVariableOpReadVariableOp,bidirectional/backward_gru/gru_cell_2/kernel*
_output_shapes

:*
dtype0
È
6bidirectional/backward_gru/gru_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*G
shared_name86bidirectional/backward_gru/gru_cell_2/recurrent_kernel
Á
Jbidirectional/backward_gru/gru_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp6bidirectional/backward_gru/gru_cell_2/recurrent_kernel*
_output_shapes

:
*
dtype0
°
*bidirectional/backward_gru/gru_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*bidirectional/backward_gru/gru_cell_2/bias
©
>bidirectional/backward_gru/gru_cell_2/bias/Read/ReadVariableOpReadVariableOp*bidirectional/backward_gru/gru_cell_2/bias*
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

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
À
2Adam/bidirectional/forward_gru/gru_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/bidirectional/forward_gru/gru_cell_1/kernel/m
¹
FAdam/bidirectional/forward_gru/gru_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/bidirectional/forward_gru/gru_cell_1/kernel/m*
_output_shapes

:*
dtype0
Ô
<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*M
shared_name><Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/m
Í
PAdam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/m*
_output_shapes

:
*
dtype0
¼
0Adam/bidirectional/forward_gru/gru_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/bidirectional/forward_gru/gru_cell_1/bias/m
µ
DAdam/bidirectional/forward_gru/gru_cell_1/bias/m/Read/ReadVariableOpReadVariableOp0Adam/bidirectional/forward_gru/gru_cell_1/bias/m*
_output_shapes

:*
dtype0
Â
3Adam/bidirectional/backward_gru/gru_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53Adam/bidirectional/backward_gru/gru_cell_2/kernel/m
»
GAdam/bidirectional/backward_gru/gru_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp3Adam/bidirectional/backward_gru/gru_cell_2/kernel/m*
_output_shapes

:*
dtype0
Ö
=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*N
shared_name?=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/m
Ï
QAdam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/m*
_output_shapes

:
*
dtype0
¾
1Adam/bidirectional/backward_gru/gru_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/bidirectional/backward_gru/gru_cell_2/bias/m
·
EAdam/bidirectional/backward_gru/gru_cell_2/bias/m/Read/ReadVariableOpReadVariableOp1Adam/bidirectional/backward_gru/gru_cell_2/bias/m*
_output_shapes

:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
À
2Adam/bidirectional/forward_gru/gru_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/bidirectional/forward_gru/gru_cell_1/kernel/v
¹
FAdam/bidirectional/forward_gru/gru_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/bidirectional/forward_gru/gru_cell_1/kernel/v*
_output_shapes

:*
dtype0
Ô
<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*M
shared_name><Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/v
Í
PAdam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/v*
_output_shapes

:
*
dtype0
¼
0Adam/bidirectional/forward_gru/gru_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/bidirectional/forward_gru/gru_cell_1/bias/v
µ
DAdam/bidirectional/forward_gru/gru_cell_1/bias/v/Read/ReadVariableOpReadVariableOp0Adam/bidirectional/forward_gru/gru_cell_1/bias/v*
_output_shapes

:*
dtype0
Â
3Adam/bidirectional/backward_gru/gru_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53Adam/bidirectional/backward_gru/gru_cell_2/kernel/v
»
GAdam/bidirectional/backward_gru/gru_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp3Adam/bidirectional/backward_gru/gru_cell_2/kernel/v*
_output_shapes

:*
dtype0
Ö
=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*N
shared_name?=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/v
Ï
QAdam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/v*
_output_shapes

:
*
dtype0
¾
1Adam/bidirectional/backward_gru/gru_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31Adam/bidirectional/backward_gru/gru_cell_2/bias/v
·
EAdam/bidirectional/backward_gru/gru_cell_2/bias/v/Read/ReadVariableOpReadVariableOp1Adam/bidirectional/backward_gru/gru_cell_2/bias/v*
_output_shapes

:*
dtype0

NoOpNoOp
O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÙN
valueÏNBÌN BÅN
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
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ke
VARIABLE_VALUE+bidirectional/forward_gru/gru_cell_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5bidirectional/forward_gru/gru_cell_1/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)bidirectional/forward_gru/gru_cell_1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,bidirectional/backward_gru/gru_cell_2/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6bidirectional/backward_gru/gru_cell_2/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*bidirectional/backward_gru/gru_cell_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/bidirectional/forward_gru/gru_cell_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/bidirectional/forward_gru/gru_cell_1/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/bidirectional/backward_gru/gru_cell_2/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/bidirectional/backward_gru/gru_cell_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/bidirectional/forward_gru/gru_cell_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/bidirectional/forward_gru/gru_cell_1/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/bidirectional/backward_gru/gru_cell_2/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/bidirectional/backward_gru/gru_cell_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_bidirectional_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
ª
StatefulPartitionedCallStatefulPartitionedCall#serving_default_bidirectional_input)bidirectional/forward_gru/gru_cell_1/bias+bidirectional/forward_gru/gru_cell_1/kernel5bidirectional/forward_gru/gru_cell_1/recurrent_kernel*bidirectional/backward_gru/gru_cell_2/bias,bidirectional/backward_gru/gru_cell_2/kernel6bidirectional/backward_gru/gru_cell_2/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/bias*
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
#__inference_signature_wrapper_19835
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
É
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?bidirectional/forward_gru/gru_cell_1/kernel/Read/ReadVariableOpIbidirectional/forward_gru/gru_cell_1/recurrent_kernel/Read/ReadVariableOp=bidirectional/forward_gru/gru_cell_1/bias/Read/ReadVariableOp@bidirectional/backward_gru/gru_cell_2/kernel/Read/ReadVariableOpJbidirectional/backward_gru/gru_cell_2/recurrent_kernel/Read/ReadVariableOp>bidirectional/backward_gru/gru_cell_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOpFAdam/bidirectional/forward_gru/gru_cell_1/kernel/m/Read/ReadVariableOpPAdam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOpDAdam/bidirectional/forward_gru/gru_cell_1/bias/m/Read/ReadVariableOpGAdam/bidirectional/backward_gru/gru_cell_2/kernel/m/Read/ReadVariableOpQAdam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/m/Read/ReadVariableOpEAdam/bidirectional/backward_gru/gru_cell_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpFAdam/bidirectional/forward_gru/gru_cell_1/kernel/v/Read/ReadVariableOpPAdam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpDAdam/bidirectional/forward_gru/gru_cell_1/bias/v/Read/ReadVariableOpGAdam/bidirectional/backward_gru/gru_cell_2/kernel/v/Read/ReadVariableOpQAdam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/v/Read/ReadVariableOpEAdam/bidirectional/backward_gru/gru_cell_2/bias/v/Read/ReadVariableOpConst*4
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
__inference__traced_save_23017
¸
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate+bidirectional/forward_gru/gru_cell_1/kernel5bidirectional/forward_gru/gru_cell_1/recurrent_kernel)bidirectional/forward_gru/gru_cell_1/bias,bidirectional/backward_gru/gru_cell_2/kernel6bidirectional/backward_gru/gru_cell_2/recurrent_kernel*bidirectional/backward_gru/gru_cell_2/biastotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m2Adam/bidirectional/forward_gru/gru_cell_1/kernel/m<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/m0Adam/bidirectional/forward_gru/gru_cell_1/bias/m3Adam/bidirectional/backward_gru/gru_cell_2/kernel/m=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/m1Adam/bidirectional/backward_gru/gru_cell_2/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v2Adam/bidirectional/forward_gru/gru_cell_1/kernel/v<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/v0Adam/bidirectional/forward_gru/gru_cell_1/bias/v3Adam/bidirectional/backward_gru/gru_cell_2/kernel/v=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/v1Adam/bidirectional/backward_gru/gru_cell_2/bias/v*3
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
!__inference__traced_restore_23144´8
	

-__inference_bidirectional_layer_call_fn_19915

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18869o
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
ÓÐ
ó
H__inference_bidirectional_layer_call_and_return_conditional_losses_21187

inputs@
.forward_gru_gru_cell_1_readvariableop_resource:G
5forward_gru_gru_cell_1_matmul_readvariableop_resource:I
7forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
A
/backward_gru_gru_cell_2_readvariableop_resource:H
6backward_gru_gru_cell_2_matmul_readvariableop_resource:J
8backward_gru_gru_cell_2_matmul_1_readvariableop_resource:

identity¢-backward_gru/gru_cell_2/MatMul/ReadVariableOp¢/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢&backward_gru/gru_cell_2/ReadVariableOp¢backward_gru/while¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢,forward_gru/gru_cell_1/MatMul/ReadVariableOp¢.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢%forward_gru/gru_cell_1/ReadVariableOp¢forward_gru/whileG
forward_gru/ShapeShapeinputs*
T0*
_output_shapes
:i
forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_sliceStridedSliceforward_gru/Shape:output:0(forward_gru/strided_slice/stack:output:0*forward_gru/strided_slice/stack_1:output:0*forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru/zeros/packedPack"forward_gru/strided_slice:output:0#forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru/zerosFill!forward_gru/zeros/packed:output:0 forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru/transpose	Transposeinputs#forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
forward_gru/Shape_1Shapeforward_gru/transpose:y:0*
T0*
_output_shapes
:k
!forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_slice_1StridedSliceforward_gru/Shape_1:output:0*forward_gru/strided_slice_1/stack:output:0,forward_gru/strided_slice_1/stack_1:output:0,forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
forward_gru/TensorArrayV2TensorListReserve0forward_gru/TensorArrayV2/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
3forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru/transpose:y:0Jforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
forward_gru/strided_slice_2StridedSliceforward_gru/transpose:y:0*forward_gru/strided_slice_2/stack:output:0,forward_gru/strided_slice_2/stack_1:output:0,forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
%forward_gru/gru_cell_1/ReadVariableOpReadVariableOp.forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0
forward_gru/gru_cell_1/unstackUnpack-forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¢
,forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
forward_gru/gru_cell_1/MatMulMatMul$forward_gru/strided_slice_2:output:04forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
forward_gru/gru_cell_1/BiasAddBiasAdd'forward_gru/gru_cell_1/MatMul:product:0'forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
forward_gru/gru_cell_1/splitSplit/forward_gru/gru_cell_1/split/split_dim:output:0'forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¦
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¯
forward_gru/gru_cell_1/MatMul_1MatMulforward_gru/zeros:output:06forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 forward_gru/gru_cell_1/BiasAdd_1BiasAdd)forward_gru/gru_cell_1/MatMul_1:product:0'forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿs
(forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
forward_gru/gru_cell_1/split_1SplitV)forward_gru/gru_cell_1/BiasAdd_1:output:0%forward_gru/gru_cell_1/Const:output:01forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¥
forward_gru/gru_cell_1/addAddV2%forward_gru/gru_cell_1/split:output:0'forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru/gru_cell_1/SigmoidSigmoidforward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
forward_gru/gru_cell_1/add_1AddV2%forward_gru/gru_cell_1/split:output:1'forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru/gru_cell_1/Sigmoid_1Sigmoid forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
forward_gru/gru_cell_1/mulMul$forward_gru/gru_cell_1/Sigmoid_1:y:0'forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_2AddV2%forward_gru/gru_cell_1/split:output:2forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
forward_gru/gru_cell_1/ReluRelu forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/mul_1Mul"forward_gru/gru_cell_1/Sigmoid:y:0forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
forward_gru/gru_cell_1/subSub%forward_gru/gru_cell_1/sub/x:output:0"forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
forward_gru/gru_cell_1/mul_2Mulforward_gru/gru_cell_1/sub:z:0)forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_3AddV2 forward_gru/gru_cell_1/mul_1:z:0 forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ü
forward_gru/TensorArrayV2_1TensorListReserve2forward_gru/TensorArrayV2_1/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
forward_gru/whileWhile'forward_gru/while/loop_counter:output:0-forward_gru/while/maximum_iterations:output:0forward_gru/time:output:0$forward_gru/TensorArrayV2_1:handle:0forward_gru/zeros:output:0$forward_gru/strided_slice_1:output:0Cforward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0.forward_gru_gru_cell_1_readvariableop_resource5forward_gru_gru_cell_1_matmul_readvariableop_resource7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *(
body R
forward_gru_while_body_20933*(
cond R
forward_gru_while_cond_20932*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
<forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   æ
.forward_gru/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru/while:output:3Eforward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0t
!forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
forward_gru/strided_slice_3StridedSlice7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0*forward_gru/strided_slice_3/stack:output:0,forward_gru/strided_slice_3/stack_1:output:0,forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskq
forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
forward_gru/transpose_1	Transpose7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0%forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    H
backward_gru/ShapeShapeinputs*
T0*
_output_shapes
:j
 backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_sliceStridedSlicebackward_gru/Shape:output:0)backward_gru/strided_slice/stack:output:0+backward_gru/strided_slice/stack_1:output:0+backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

backward_gru/zeros/packedPack#backward_gru/strided_slice:output:0$backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru/zerosFill"backward_gru/zeros/packed:output:0!backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru/transpose	Transposeinputs$backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
backward_gru/Shape_1Shapebackward_gru/transpose:y:0*
T0*
_output_shapes
:l
"backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_slice_1StridedSlicebackward_gru/Shape_1:output:0+backward_gru/strided_slice_1/stack:output:0-backward_gru/strided_slice_1/stack_1:output:0-backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
backward_gru/TensorArrayV2TensorListReserve1backward_gru/TensorArrayV2/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_gru/ReverseV2	ReverseV2backward_gru/transpose:y:0$backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
4backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbackward_gru/ReverseV2:output:0Kbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
backward_gru/strided_slice_2StridedSlicebackward_gru/transpose:y:0+backward_gru/strided_slice_2/stack:output:0-backward_gru/strided_slice_2/stack_1:output:0-backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
&backward_gru/gru_cell_2/ReadVariableOpReadVariableOp/backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0
backward_gru/gru_cell_2/unstackUnpack.backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¤
-backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
backward_gru/gru_cell_2/MatMulMatMul%backward_gru/strided_slice_2:output:05backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
backward_gru/gru_cell_2/BiasAddBiasAdd(backward_gru/gru_cell_2/MatMul:product:0(backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
backward_gru/gru_cell_2/splitSplit0backward_gru/gru_cell_2/split/split_dim:output:0(backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0²
 backward_gru/gru_cell_2/MatMul_1MatMulbackward_gru/zeros:output:07backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!backward_gru/gru_cell_2/BiasAdd_1BiasAdd*backward_gru/gru_cell_2/MatMul_1:product:0(backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿt
)backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
backward_gru/gru_cell_2/split_1SplitV*backward_gru/gru_cell_2/BiasAdd_1:output:0&backward_gru/gru_cell_2/Const:output:02backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
backward_gru/gru_cell_2/addAddV2&backward_gru/gru_cell_2/split:output:0(backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru/gru_cell_2/SigmoidSigmoidbackward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
backward_gru/gru_cell_2/add_1AddV2&backward_gru/gru_cell_2/split:output:1(backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru/gru_cell_2/Sigmoid_1Sigmoid!backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
backward_gru/gru_cell_2/mulMul%backward_gru/gru_cell_2/Sigmoid_1:y:0(backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
backward_gru/gru_cell_2/add_2AddV2&backward_gru/gru_cell_2/split:output:2backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
backward_gru/gru_cell_2/ReluRelu!backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/mul_1Mul#backward_gru/gru_cell_2/Sigmoid:y:0backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
backward_gru/gru_cell_2/subSub&backward_gru/gru_cell_2/sub/x:output:0#backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
backward_gru/gru_cell_2/mul_2Mulbackward_gru/gru_cell_2/sub:z:0*backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/add_3AddV2!backward_gru/gru_cell_2/mul_1:z:0!backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
*backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ß
backward_gru/TensorArrayV2_1TensorListReserve3backward_gru/TensorArrayV2_1/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
backward_gru/whileWhile(backward_gru/while/loop_counter:output:0.backward_gru/while/maximum_iterations:output:0backward_gru/time:output:0%backward_gru/TensorArrayV2_1:handle:0backward_gru/zeros:output:0%backward_gru/strided_slice_1:output:0Dbackward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/backward_gru_gru_cell_2_readvariableop_resource6backward_gru_gru_cell_2_matmul_readvariableop_resource8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *)
body!R
backward_gru_while_body_21084*)
cond!R
backward_gru_while_cond_21083*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
=backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   é
/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru/while:output:3Fbackward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
"backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
backward_gru/strided_slice_3StridedSlice8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0+backward_gru/strided_slice_3/stack:output:0-backward_gru/strided_slice_3/stack_1:output:0-backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskr
backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
backward_gru/transpose_1	Transpose8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0&backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2$forward_gru/strided_slice_3:output:0%backward_gru/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Å
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp.^backward_gru/gru_cell_2/MatMul/ReadVariableOp0^backward_gru/gru_cell_2/MatMul_1/ReadVariableOp'^backward_gru/gru_cell_2/ReadVariableOp^backward_gru/whileO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp-^forward_gru/gru_cell_1/MatMul/ReadVariableOp/^forward_gru/gru_cell_1/MatMul_1/ReadVariableOp&^forward_gru/gru_cell_1/ReadVariableOp^forward_gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_gru/gru_cell_2/MatMul/ReadVariableOp-backward_gru/gru_cell_2/MatMul/ReadVariableOp2b
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2P
&backward_gru/gru_cell_2/ReadVariableOp&backward_gru/gru_cell_2/ReadVariableOp2(
backward_gru/whilebackward_gru/while2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2\
,forward_gru/gru_cell_1/MatMul/ReadVariableOp,forward_gru/gru_cell_1/MatMul/ReadVariableOp2`
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2N
%forward_gru/gru_cell_1/ReadVariableOp%forward_gru/gru_cell_1/ReadVariableOp2&
forward_gru/whileforward_gru/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_22028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22028___redundant_placeholder03
/while_while_cond_22028___redundant_placeholder13
/while_while_cond_22028___redundant_placeholder23
/while_while_cond_22028___redundant_placeholder3
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
ð*
Ñ
E__inference_sequential_layer_call_and_return_conditional_losses_18489

inputs%
bidirectional_18429:%
bidirectional_18431:%
bidirectional_18433:
%
bidirectional_18435:%
bidirectional_18437:%
bidirectional_18439:

dense_18454:
dense_18456:
dense_1_18471:
dense_1_18473:
identity¢%bidirectional/StatefulPartitionedCall¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÝ
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_18429bidirectional_18431bidirectional_18433bidirectional_18435bidirectional_18437bidirectional_18439*
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18428
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_18454dense_18456*
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
GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18453
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_18471dense_1_18473*
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
B__inference_dense_1_layer_call_and_return_conditional_losses_18470¡
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_18431*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_18437*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp&^bidirectional/StatefulPartitionedCallO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
,__inference_backward_gru_layer_call_fn_21941
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallë
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
GPU 2J 8 *P
fKRI
G__inference_backward_gru_layer_call_and_return_conditional_losses_17311o
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
ïÑ
õ
H__inference_bidirectional_layer_call_and_return_conditional_losses_20233
inputs_0@
.forward_gru_gru_cell_1_readvariableop_resource:G
5forward_gru_gru_cell_1_matmul_readvariableop_resource:I
7forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
A
/backward_gru_gru_cell_2_readvariableop_resource:H
6backward_gru_gru_cell_2_matmul_readvariableop_resource:J
8backward_gru_gru_cell_2_matmul_1_readvariableop_resource:

identity¢-backward_gru/gru_cell_2/MatMul/ReadVariableOp¢/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢&backward_gru/gru_cell_2/ReadVariableOp¢backward_gru/while¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢,forward_gru/gru_cell_1/MatMul/ReadVariableOp¢.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢%forward_gru/gru_cell_1/ReadVariableOp¢forward_gru/whileI
forward_gru/ShapeShapeinputs_0*
T0*
_output_shapes
:i
forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_sliceStridedSliceforward_gru/Shape:output:0(forward_gru/strided_slice/stack:output:0*forward_gru/strided_slice/stack_1:output:0*forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru/zeros/packedPack"forward_gru/strided_slice:output:0#forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru/zerosFill!forward_gru/zeros/packed:output:0 forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru/transpose	Transposeinputs_0#forward_gru/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ\
forward_gru/Shape_1Shapeforward_gru/transpose:y:0*
T0*
_output_shapes
:k
!forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_slice_1StridedSliceforward_gru/Shape_1:output:0*forward_gru/strided_slice_1/stack:output:0,forward_gru/strided_slice_1/stack_1:output:0,forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
forward_gru/TensorArrayV2TensorListReserve0forward_gru/TensorArrayV2/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
3forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru/transpose:y:0Jforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
forward_gru/strided_slice_2StridedSliceforward_gru/transpose:y:0*forward_gru/strided_slice_2/stack:output:0,forward_gru/strided_slice_2/stack_1:output:0,forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
%forward_gru/gru_cell_1/ReadVariableOpReadVariableOp.forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0
forward_gru/gru_cell_1/unstackUnpack-forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¢
,forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
forward_gru/gru_cell_1/MatMulMatMul$forward_gru/strided_slice_2:output:04forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
forward_gru/gru_cell_1/BiasAddBiasAdd'forward_gru/gru_cell_1/MatMul:product:0'forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
forward_gru/gru_cell_1/splitSplit/forward_gru/gru_cell_1/split/split_dim:output:0'forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¦
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¯
forward_gru/gru_cell_1/MatMul_1MatMulforward_gru/zeros:output:06forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 forward_gru/gru_cell_1/BiasAdd_1BiasAdd)forward_gru/gru_cell_1/MatMul_1:product:0'forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿs
(forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
forward_gru/gru_cell_1/split_1SplitV)forward_gru/gru_cell_1/BiasAdd_1:output:0%forward_gru/gru_cell_1/Const:output:01forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¥
forward_gru/gru_cell_1/addAddV2%forward_gru/gru_cell_1/split:output:0'forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru/gru_cell_1/SigmoidSigmoidforward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
forward_gru/gru_cell_1/add_1AddV2%forward_gru/gru_cell_1/split:output:1'forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru/gru_cell_1/Sigmoid_1Sigmoid forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
forward_gru/gru_cell_1/mulMul$forward_gru/gru_cell_1/Sigmoid_1:y:0'forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_2AddV2%forward_gru/gru_cell_1/split:output:2forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
forward_gru/gru_cell_1/ReluRelu forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/mul_1Mul"forward_gru/gru_cell_1/Sigmoid:y:0forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
forward_gru/gru_cell_1/subSub%forward_gru/gru_cell_1/sub/x:output:0"forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
forward_gru/gru_cell_1/mul_2Mulforward_gru/gru_cell_1/sub:z:0)forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_3AddV2 forward_gru/gru_cell_1/mul_1:z:0 forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ü
forward_gru/TensorArrayV2_1TensorListReserve2forward_gru/TensorArrayV2_1/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
forward_gru/whileWhile'forward_gru/while/loop_counter:output:0-forward_gru/while/maximum_iterations:output:0forward_gru/time:output:0$forward_gru/TensorArrayV2_1:handle:0forward_gru/zeros:output:0$forward_gru/strided_slice_1:output:0Cforward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0.forward_gru_gru_cell_1_readvariableop_resource5forward_gru_gru_cell_1_matmul_readvariableop_resource7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *(
body R
forward_gru_while_body_19979*(
cond R
forward_gru_while_cond_19978*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
<forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ï
.forward_gru/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru/while:output:3Eforward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0t
!forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
forward_gru/strided_slice_3StridedSlice7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0*forward_gru/strided_slice_3/stack:output:0,forward_gru/strided_slice_3/stack_1:output:0,forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskq
forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
forward_gru/transpose_1	Transpose7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0%forward_gru/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
g
forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    J
backward_gru/ShapeShapeinputs_0*
T0*
_output_shapes
:j
 backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_sliceStridedSlicebackward_gru/Shape:output:0)backward_gru/strided_slice/stack:output:0+backward_gru/strided_slice/stack_1:output:0+backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

backward_gru/zeros/packedPack#backward_gru/strided_slice:output:0$backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru/zerosFill"backward_gru/zeros/packed:output:0!backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru/transpose	Transposeinputs_0$backward_gru/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
backward_gru/Shape_1Shapebackward_gru/transpose:y:0*
T0*
_output_shapes
:l
"backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_slice_1StridedSlicebackward_gru/Shape_1:output:0+backward_gru/strided_slice_1/stack:output:0-backward_gru/strided_slice_1/stack_1:output:0-backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
backward_gru/TensorArrayV2TensorListReserve1backward_gru/TensorArrayV2/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ­
backward_gru/ReverseV2	ReverseV2backward_gru/transpose:y:0$backward_gru/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Bbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
4backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbackward_gru/ReverseV2:output:0Kbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
backward_gru/strided_slice_2StridedSlicebackward_gru/transpose:y:0+backward_gru/strided_slice_2/stack:output:0-backward_gru/strided_slice_2/stack_1:output:0-backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
&backward_gru/gru_cell_2/ReadVariableOpReadVariableOp/backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0
backward_gru/gru_cell_2/unstackUnpack.backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¤
-backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
backward_gru/gru_cell_2/MatMulMatMul%backward_gru/strided_slice_2:output:05backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
backward_gru/gru_cell_2/BiasAddBiasAdd(backward_gru/gru_cell_2/MatMul:product:0(backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
backward_gru/gru_cell_2/splitSplit0backward_gru/gru_cell_2/split/split_dim:output:0(backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0²
 backward_gru/gru_cell_2/MatMul_1MatMulbackward_gru/zeros:output:07backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!backward_gru/gru_cell_2/BiasAdd_1BiasAdd*backward_gru/gru_cell_2/MatMul_1:product:0(backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿt
)backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
backward_gru/gru_cell_2/split_1SplitV*backward_gru/gru_cell_2/BiasAdd_1:output:0&backward_gru/gru_cell_2/Const:output:02backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
backward_gru/gru_cell_2/addAddV2&backward_gru/gru_cell_2/split:output:0(backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru/gru_cell_2/SigmoidSigmoidbackward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
backward_gru/gru_cell_2/add_1AddV2&backward_gru/gru_cell_2/split:output:1(backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru/gru_cell_2/Sigmoid_1Sigmoid!backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
backward_gru/gru_cell_2/mulMul%backward_gru/gru_cell_2/Sigmoid_1:y:0(backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
backward_gru/gru_cell_2/add_2AddV2&backward_gru/gru_cell_2/split:output:2backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
backward_gru/gru_cell_2/ReluRelu!backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/mul_1Mul#backward_gru/gru_cell_2/Sigmoid:y:0backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
backward_gru/gru_cell_2/subSub&backward_gru/gru_cell_2/sub/x:output:0#backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
backward_gru/gru_cell_2/mul_2Mulbackward_gru/gru_cell_2/sub:z:0*backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/add_3AddV2!backward_gru/gru_cell_2/mul_1:z:0!backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
*backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ß
backward_gru/TensorArrayV2_1TensorListReserve3backward_gru/TensorArrayV2_1/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
backward_gru/whileWhile(backward_gru/while/loop_counter:output:0.backward_gru/while/maximum_iterations:output:0backward_gru/time:output:0%backward_gru/TensorArrayV2_1:handle:0backward_gru/zeros:output:0%backward_gru/strided_slice_1:output:0Dbackward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/backward_gru_gru_cell_2_readvariableop_resource6backward_gru_gru_cell_2_matmul_readvariableop_resource8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *)
body!R
backward_gru_while_body_20130*)
cond!R
backward_gru_while_cond_20129*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
=backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ò
/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru/while:output:3Fbackward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
"backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
backward_gru/strided_slice_3StridedSlice8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0+backward_gru/strided_slice_3/stack:output:0-backward_gru/strided_slice_3/stack_1:output:0-backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskr
backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Æ
backward_gru/transpose_1	Transpose8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0&backward_gru/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
h
backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2$forward_gru/strided_slice_3:output:0%backward_gru/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Å
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp.^backward_gru/gru_cell_2/MatMul/ReadVariableOp0^backward_gru/gru_cell_2/MatMul_1/ReadVariableOp'^backward_gru/gru_cell_2/ReadVariableOp^backward_gru/whileO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp-^forward_gru/gru_cell_1/MatMul/ReadVariableOp/^forward_gru/gru_cell_1/MatMul_1/ReadVariableOp&^forward_gru/gru_cell_1/ReadVariableOp^forward_gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_gru/gru_cell_2/MatMul/ReadVariableOp-backward_gru/gru_cell_2/MatMul/ReadVariableOp2b
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2P
&backward_gru/gru_cell_2/ReadVariableOp&backward_gru/gru_cell_2/ReadVariableOp2(
backward_gru/whilebackward_gru/while2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2\
,forward_gru/gru_cell_1/MatMul/ReadVariableOp,forward_gru/gru_cell_1/MatMul/ReadVariableOp2`
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2N
%forward_gru/gru_cell_1/ReadVariableOp%forward_gru/gru_cell_1/ReadVariableOp2&
forward_gru/whileforward_gru/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÍZ
Õ
G__inference_backward_gru_layer_call_and_return_conditional_losses_22446

inputs4
"gru_cell_2_readvariableop_resource:;
)gru_cell_2_matmul_readvariableop_resource:=
+gru_cell_2_matmul_1_readvariableop_resource:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢while;
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
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
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
while_body_22351*
condR
while_cond_22350*8
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
 *    ¸
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs



forward_gru_while_cond_181734
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_26
2forward_gru_while_less_forward_gru_strided_slice_1K
Gforward_gru_while_forward_gru_while_cond_18173___redundant_placeholder0K
Gforward_gru_while_forward_gru_while_cond_18173___redundant_placeholder1K
Gforward_gru_while_forward_gru_while_cond_18173___redundant_placeholder2K
Gforward_gru_while_forward_gru_while_cond_18173___redundant_placeholder3
forward_gru_while_identity

forward_gru/while/LessLessforward_gru_while_placeholder2forward_gru_while_less_forward_gru_strided_slice_1*
T0*
_output_shapes
: c
forward_gru/while/IdentityIdentityforward_gru/while/Less:z:0*
T0
*
_output_shapes
: "A
forward_gru_while_identity#forward_gru/while/Identity:output:0*(
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
while_cond_16874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_16874___redundant_placeholder03
/while_while_cond_16874___redundant_placeholder13
/while_while_cond_16874___redundant_placeholder23
/while_while_cond_16874___redundant_placeholder3
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
»(
¦
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_22731

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpf
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
¬
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
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
Ù
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:O K
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
­

Ö
*__inference_gru_cell_2_layer_call_fn_22776

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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17187o
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
ÍZ
Õ
G__inference_backward_gru_layer_call_and_return_conditional_losses_22607

inputs4
"gru_cell_2_readvariableop_resource:;
)gru_cell_2_matmul_readvariableop_resource:=
+gru_cell_2_matmul_1_readvariableop_resource:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢while;
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
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
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
while_body_22512*
condR
while_cond_22511*8
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
 *    ¸
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
]

*bidirectional_forward_gru_while_body_19540P
Lbidirectional_forward_gru_while_bidirectional_forward_gru_while_loop_counterV
Rbidirectional_forward_gru_while_bidirectional_forward_gru_while_maximum_iterations/
+bidirectional_forward_gru_while_placeholder1
-bidirectional_forward_gru_while_placeholder_11
-bidirectional_forward_gru_while_placeholder_2O
Kbidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1_0
bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0V
Dbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0:]
Kbidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:_
Mbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:
,
(bidirectional_forward_gru_while_identity.
*bidirectional_forward_gru_while_identity_1.
*bidirectional_forward_gru_while_identity_2.
*bidirectional_forward_gru_while_identity_3.
*bidirectional_forward_gru_while_identity_4M
Ibidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1
bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensorT
Bbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource:[
Ibidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource:]
Kbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢Bbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp¢
Qbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ©
Cbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0+bidirectional_forward_gru_while_placeholderZbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¾
9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOpDbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
2bidirectional/forward_gru/while/gru_cell_1/unstackUnpackAbidirectional/forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÌ
@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOpKbidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
1bidirectional/forward_gru/while/gru_cell_1/MatMulMatMulJbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0Hbidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
2bidirectional/forward_gru/while/gru_cell_1/BiasAddBiasAdd;bidirectional/forward_gru/while/gru_cell_1/MatMul:product:0;bidirectional/forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:bidirectional/forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
0bidirectional/forward_gru/while/gru_cell_1/splitSplitCbidirectional/forward_gru/while/gru_cell_1/split/split_dim:output:0;bidirectional/forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÐ
Bbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpMbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0ê
3bidirectional/forward_gru/while/gru_cell_1/MatMul_1MatMul-bidirectional_forward_gru_while_placeholder_2Jbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
4bidirectional/forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd=bidirectional/forward_gru/while/gru_cell_1/MatMul_1:product:0;bidirectional/forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0bidirectional/forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
<bidirectional/forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
2bidirectional/forward_gru/while/gru_cell_1/split_1SplitV=bidirectional/forward_gru/while/gru_cell_1/BiasAdd_1:output:09bidirectional/forward_gru/while/gru_cell_1/Const:output:0Ebidirectional/forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitá
.bidirectional/forward_gru/while/gru_cell_1/addAddV29bidirectional/forward_gru/while/gru_cell_1/split:output:0;bidirectional/forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2bidirectional/forward_gru/while/gru_cell_1/SigmoidSigmoid2bidirectional/forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ã
0bidirectional/forward_gru/while/gru_cell_1/add_1AddV29bidirectional/forward_gru/while/gru_cell_1/split:output:1;bidirectional/forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
4bidirectional/forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid4bidirectional/forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
.bidirectional/forward_gru/while/gru_cell_1/mulMul8bidirectional/forward_gru/while/gru_cell_1/Sigmoid_1:y:0;bidirectional/forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
0bidirectional/forward_gru/while/gru_cell_1/add_2AddV29bidirectional/forward_gru/while/gru_cell_1/split:output:22bidirectional/forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/bidirectional/forward_gru/while/gru_cell_1/ReluRelu4bidirectional/forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ð
0bidirectional/forward_gru/while/gru_cell_1/mul_1Mul6bidirectional/forward_gru/while/gru_cell_1/Sigmoid:y:0-bidirectional_forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
0bidirectional/forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ú
.bidirectional/forward_gru/while/gru_cell_1/subSub9bidirectional/forward_gru/while/gru_cell_1/sub/x:output:06bidirectional/forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ü
0bidirectional/forward_gru/while/gru_cell_1/mul_2Mul2bidirectional/forward_gru/while/gru_cell_1/sub:z:0=bidirectional/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
0bidirectional/forward_gru/while/gru_cell_1/add_3AddV24bidirectional/forward_gru/while/gru_cell_1/mul_1:z:04bidirectional/forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
Dbidirectional/forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-bidirectional_forward_gru_while_placeholder_1+bidirectional_forward_gru_while_placeholder4bidirectional/forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒg
%bidirectional/forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ª
#bidirectional/forward_gru/while/addAddV2+bidirectional_forward_gru_while_placeholder.bidirectional/forward_gru/while/add/y:output:0*
T0*
_output_shapes
: i
'bidirectional/forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ï
%bidirectional/forward_gru/while/add_1AddV2Lbidirectional_forward_gru_while_bidirectional_forward_gru_while_loop_counter0bidirectional/forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: §
(bidirectional/forward_gru/while/IdentityIdentity)bidirectional/forward_gru/while/add_1:z:0%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: Ò
*bidirectional/forward_gru/while/Identity_1IdentityRbidirectional_forward_gru_while_bidirectional_forward_gru_while_maximum_iterations%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: §
*bidirectional/forward_gru/while/Identity_2Identity'bidirectional/forward_gru/while/add:z:0%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: ç
*bidirectional/forward_gru/while/Identity_3IdentityTbidirectional/forward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒÅ
*bidirectional/forward_gru/while/Identity_4Identity4bidirectional/forward_gru/while/gru_cell_1/add_3:z:0%^bidirectional/forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
$bidirectional/forward_gru/while/NoOpNoOpA^bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpC^bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:^bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Ibidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1Kbidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1_0"
Kbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resourceMbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"
Ibidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resourceKbidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"
Bbidirectional_forward_gru_while_gru_cell_1_readvariableop_resourceDbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0"]
(bidirectional_forward_gru_while_identity1bidirectional/forward_gru/while/Identity:output:0"a
*bidirectional_forward_gru_while_identity_13bidirectional/forward_gru/while/Identity_1:output:0"a
*bidirectional_forward_gru_while_identity_23bidirectional/forward_gru/while/Identity_2:output:0"a
*bidirectional_forward_gru_while_identity_33bidirectional/forward_gru/while/Identity_3:output:0"a
*bidirectional_forward_gru_while_identity_43bidirectional/forward_gru/while/Identity_4:output:0"
bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensorbidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2
Bbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpBbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2v
9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp: 
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
²^
¦
+bidirectional_backward_gru_while_body_19359R
Nbidirectional_backward_gru_while_bidirectional_backward_gru_while_loop_counterX
Tbidirectional_backward_gru_while_bidirectional_backward_gru_while_maximum_iterations0
,bidirectional_backward_gru_while_placeholder2
.bidirectional_backward_gru_while_placeholder_12
.bidirectional_backward_gru_while_placeholder_2Q
Mbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1_0
bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0W
Ebidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0:^
Lbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:`
Nbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:
-
)bidirectional_backward_gru_while_identity/
+bidirectional_backward_gru_while_identity_1/
+bidirectional_backward_gru_while_identity_2/
+bidirectional_backward_gru_while_identity_3/
+bidirectional_backward_gru_while_identity_4O
Kbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1
bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensorU
Cbidirectional_backward_gru_while_gru_cell_2_readvariableop_resource:\
Jbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource:^
Lbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢Abidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢Cbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp£
Rbidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ®
Dbidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0,bidirectional_backward_gru_while_placeholder[bidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0À
:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOpEbidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0·
3bidirectional/backward_gru/while/gru_cell_2/unstackUnpackBbidirectional/backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÎ
Abidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOpLbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
2bidirectional/backward_gru/while/gru_cell_2/MatMulMatMulKbidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0Ibidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
3bidirectional/backward_gru/while/gru_cell_2/BiasAddBiasAdd<bidirectional/backward_gru/while/gru_cell_2/MatMul:product:0<bidirectional/backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;bidirectional/backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
1bidirectional/backward_gru/while/gru_cell_2/splitSplitDbidirectional/backward_gru/while/gru_cell_2/split/split_dim:output:0<bidirectional/backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÒ
Cbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0í
4bidirectional/backward_gru/while/gru_cell_2/MatMul_1MatMul.bidirectional_backward_gru_while_placeholder_2Kbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
5bidirectional/backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd>bidirectional/backward_gru/while/gru_cell_2/MatMul_1:product:0<bidirectional/backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1bidirectional/backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
=bidirectional/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
3bidirectional/backward_gru/while/gru_cell_2/split_1SplitV>bidirectional/backward_gru/while/gru_cell_2/BiasAdd_1:output:0:bidirectional/backward_gru/while/gru_cell_2/Const:output:0Fbidirectional/backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitä
/bidirectional/backward_gru/while/gru_cell_2/addAddV2:bidirectional/backward_gru/while/gru_cell_2/split:output:0<bidirectional/backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
3bidirectional/backward_gru/while/gru_cell_2/SigmoidSigmoid3bidirectional/backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
1bidirectional/backward_gru/while/gru_cell_2/add_1AddV2:bidirectional/backward_gru/while/gru_cell_2/split:output:1<bidirectional/backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
5bidirectional/backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid5bidirectional/backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
á
/bidirectional/backward_gru/while/gru_cell_2/mulMul9bidirectional/backward_gru/while/gru_cell_2/Sigmoid_1:y:0<bidirectional/backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
1bidirectional/backward_gru/while/gru_cell_2/add_2AddV2:bidirectional/backward_gru/while/gru_cell_2/split:output:23bidirectional/backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
0bidirectional/backward_gru/while/gru_cell_2/ReluRelu5bidirectional/backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ó
1bidirectional/backward_gru/while/gru_cell_2/mul_1Mul7bidirectional/backward_gru/while/gru_cell_2/Sigmoid:y:0.bidirectional_backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
1bidirectional/backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ý
/bidirectional/backward_gru/while/gru_cell_2/subSub:bidirectional/backward_gru/while/gru_cell_2/sub/x:output:07bidirectional/backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
1bidirectional/backward_gru/while/gru_cell_2/mul_2Mul3bidirectional/backward_gru/while/gru_cell_2/sub:z:0>bidirectional/backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
1bidirectional/backward_gru/while/gru_cell_2/add_3AddV25bidirectional/backward_gru/while/gru_cell_2/mul_1:z:05bidirectional/backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
Ebidirectional/backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.bidirectional_backward_gru_while_placeholder_1,bidirectional_backward_gru_while_placeholder5bidirectional/backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒh
&bidirectional/backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :­
$bidirectional/backward_gru/while/addAddV2,bidirectional_backward_gru_while_placeholder/bidirectional/backward_gru/while/add/y:output:0*
T0*
_output_shapes
: j
(bidirectional/backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ó
&bidirectional/backward_gru/while/add_1AddV2Nbidirectional_backward_gru_while_bidirectional_backward_gru_while_loop_counter1bidirectional/backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: ª
)bidirectional/backward_gru/while/IdentityIdentity*bidirectional/backward_gru/while/add_1:z:0&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: Ö
+bidirectional/backward_gru/while/Identity_1IdentityTbidirectional_backward_gru_while_bidirectional_backward_gru_while_maximum_iterations&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: ª
+bidirectional/backward_gru/while/Identity_2Identity(bidirectional/backward_gru/while/add:z:0&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: ê
+bidirectional/backward_gru/while/Identity_3IdentityUbidirectional/backward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒÈ
+bidirectional/backward_gru/while/Identity_4Identity5bidirectional/backward_gru/while/gru_cell_2/add_3:z:0&^bidirectional/backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
®
%bidirectional/backward_gru/while/NoOpNoOpB^bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpD^bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp;^bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Kbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1Mbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1_0"
Lbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resourceNbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"
Jbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resourceLbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"
Cbidirectional_backward_gru_while_gru_cell_2_readvariableop_resourceEbidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0"_
)bidirectional_backward_gru_while_identity2bidirectional/backward_gru/while/Identity:output:0"c
+bidirectional_backward_gru_while_identity_14bidirectional/backward_gru/while/Identity_1:output:0"c
+bidirectional_backward_gru_while_identity_24bidirectional/backward_gru/while/Identity_2:output:0"c
+bidirectional_backward_gru_while_identity_34bidirectional/backward_gru/while/Identity_3:output:0"c
+bidirectional_backward_gru_while_identity_44bidirectional/backward_gru/while/Identity_4:output:0"
bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensorbidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Abidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpAbidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2
Cbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpCbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2x
:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp: 
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
while_body_17558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:C
1while_gru_cell_2_matmul_readvariableop_resource_0:E
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:A
/while_gru_cell_2_matmul_readvariableop_resource:C
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
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
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp: 
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
ãL
þ

backward_gru_while_body_207666
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_25
1backward_gru_while_backward_gru_strided_slice_1_0q
mbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0I
7backward_gru_while_gru_cell_2_readvariableop_resource_0:P
>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:R
@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

backward_gru_while_identity!
backward_gru_while_identity_1!
backward_gru_while_identity_2!
backward_gru_while_identity_3!
backward_gru_while_identity_43
/backward_gru_while_backward_gru_strided_slice_1o
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensorG
5backward_gru_while_gru_cell_2_readvariableop_resource:N
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource:P
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢,backward_gru/while/gru_cell_2/ReadVariableOp
Dbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ç
6backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0backward_gru_while_placeholderMbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¤
,backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOp7backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
%backward_gru/while/gru_cell_2/unstackUnpack4backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ü
$backward_gru/while/gru_cell_2/MatMulMatMul=backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0;backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
%backward_gru/while/gru_cell_2/BiasAddBiasAdd.backward_gru/while/gru_cell_2/MatMul:product:0.backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#backward_gru/while/gru_cell_2/splitSplit6backward_gru/while/gru_cell_2/split/split_dim:output:0.backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¶
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Ã
&backward_gru/while/gru_cell_2/MatMul_1MatMul backward_gru_while_placeholder_2=backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd0backward_gru/while/gru_cell_2/MatMul_1:product:0.backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿz
/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%backward_gru/while/gru_cell_2/split_1SplitV0backward_gru/while/gru_cell_2/BiasAdd_1:output:0,backward_gru/while/gru_cell_2/Const:output:08backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
!backward_gru/while/gru_cell_2/addAddV2,backward_gru/while/gru_cell_2/split:output:0.backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%backward_gru/while/gru_cell_2/SigmoidSigmoid%backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
#backward_gru/while/gru_cell_2/add_1AddV2,backward_gru/while/gru_cell_2/split:output:1.backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid'backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
!backward_gru/while/gru_cell_2/mulMul+backward_gru/while/gru_cell_2/Sigmoid_1:y:0.backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
#backward_gru/while/gru_cell_2/add_2AddV2,backward_gru/while/gru_cell_2/split:output:2%backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"backward_gru/while/gru_cell_2/ReluRelu'backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
#backward_gru/while/gru_cell_2/mul_1Mul)backward_gru/while/gru_cell_2/Sigmoid:y:0 backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
#backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!backward_gru/while/gru_cell_2/subSub,backward_gru/while/gru_cell_2/sub/x:output:0)backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
#backward_gru/while/gru_cell_2/mul_2Mul%backward_gru/while/gru_cell_2/sub:z:00backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
#backward_gru/while/gru_cell_2/add_3AddV2'backward_gru/while/gru_cell_2/mul_1:z:0'backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
7backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem backward_gru_while_placeholder_1backward_gru_while_placeholder'backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒZ
backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/addAddV2backward_gru_while_placeholder!backward_gru/while/add/y:output:0*
T0*
_output_shapes
: \
backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/add_1AddV22backward_gru_while_backward_gru_while_loop_counter#backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru/while/IdentityIdentitybackward_gru/while/add_1:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_1Identity8backward_gru_while_backward_gru_while_maximum_iterations^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_2Identitybackward_gru/while/add:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: À
backward_gru/while/Identity_3IdentityGbackward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
backward_gru/while/Identity_4Identity'backward_gru/while/gru_cell_2/add_3:z:0^backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
backward_gru/while/NoOpNoOp4^backward_gru/while/gru_cell_2/MatMul/ReadVariableOp6^backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp-^backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/backward_gru_while_backward_gru_strided_slice_11backward_gru_while_backward_gru_strided_slice_1_0"
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"~
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"p
5backward_gru_while_gru_cell_2_readvariableop_resource7backward_gru_while_gru_cell_2_readvariableop_resource_0"C
backward_gru_while_identity$backward_gru/while/Identity:output:0"G
backward_gru_while_identity_1&backward_gru/while/Identity_1:output:0"G
backward_gru_while_identity_2&backward_gru/while/Identity_2:output:0"G
backward_gru_while_identity_3&backward_gru/while/Identity_3:output:0"G
backward_gru_while_identity_4&backward_gru/while/Identity_4:output:0"Ü
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensormbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2j
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2n
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2\
,backward_gru/while/gru_cell_2/ReadVariableOp,backward_gru/while/gru_cell_2/ReadVariableOp: 
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
§Z
×
G__inference_backward_gru_layer_call_and_return_conditional_losses_22285
inputs_04
"gru_cell_2_readvariableop_resource:;
)gru_cell_2_matmul_readvariableop_resource:=
+gru_cell_2_matmul_1_readvariableop_resource:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢while=
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
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
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
while_body_22190*
condR
while_cond_22189*8
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
 *    ¸
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


ó
B__inference_dense_1_layer_call_and_return_conditional_losses_21227

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
¿'
¶
H__inference_bidirectional_layer_call_and_return_conditional_losses_18089

inputs#
forward_gru_18060:#
forward_gru_18062:#
forward_gru_18064:
$
backward_gru_18067:$
backward_gru_18069:$
backward_gru_18071:

identity¢$backward_gru/StatefulPartitionedCall¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢#forward_gru/StatefulPartitionedCall
#forward_gru/StatefulPartitionedCallStatefulPartitionedCallinputsforward_gru_18060forward_gru_18062forward_gru_18064*
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
GPU 2J 8 *O
fJRH
F__inference_forward_gru_layer_call_and_return_conditional_losses_18046
$backward_gru/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_gru_18067backward_gru_18069backward_gru_18071*
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
GPU 2J 8 *P
fKRI
G__inference_backward_gru_layer_call_and_return_conditional_losses_17865M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :À
concatConcatV2,forward_gru/StatefulPartitionedCall:output:0-backward_gru/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpforward_gru_18062*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¡
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbackward_gru_18069*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp%^backward_gru/StatefulPartitionedCallO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp$^forward_gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$backward_gru/StatefulPartitionedCall$backward_gru/StatefulPartitionedCall2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2J
#forward_gru/StatefulPartitionedCall#forward_gru/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
¡
while_body_16875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_1_16897_0:*
while_gru_cell_1_16899_0:*
while_gru_cell_1_16901_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_1_16897:(
while_gru_cell_1_16899:(
while_gru_cell_1_16901:
¢(while/gru_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_16897_0while_gru_cell_1_16899_0while_gru_cell_1_16901_0*
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16823Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_1_16897while_gru_cell_1_16897_0"2
while_gru_cell_1_16899while_gru_cell_1_16899_0"2
while_gru_cell_1_16901while_gru_cell_1_16901_0")
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
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 
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
ÓÐ
ó
H__inference_bidirectional_layer_call_and_return_conditional_losses_18428

inputs@
.forward_gru_gru_cell_1_readvariableop_resource:G
5forward_gru_gru_cell_1_matmul_readvariableop_resource:I
7forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
A
/backward_gru_gru_cell_2_readvariableop_resource:H
6backward_gru_gru_cell_2_matmul_readvariableop_resource:J
8backward_gru_gru_cell_2_matmul_1_readvariableop_resource:

identity¢-backward_gru/gru_cell_2/MatMul/ReadVariableOp¢/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢&backward_gru/gru_cell_2/ReadVariableOp¢backward_gru/while¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢,forward_gru/gru_cell_1/MatMul/ReadVariableOp¢.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢%forward_gru/gru_cell_1/ReadVariableOp¢forward_gru/whileG
forward_gru/ShapeShapeinputs*
T0*
_output_shapes
:i
forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_sliceStridedSliceforward_gru/Shape:output:0(forward_gru/strided_slice/stack:output:0*forward_gru/strided_slice/stack_1:output:0*forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru/zeros/packedPack"forward_gru/strided_slice:output:0#forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru/zerosFill!forward_gru/zeros/packed:output:0 forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru/transpose	Transposeinputs#forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
forward_gru/Shape_1Shapeforward_gru/transpose:y:0*
T0*
_output_shapes
:k
!forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_slice_1StridedSliceforward_gru/Shape_1:output:0*forward_gru/strided_slice_1/stack:output:0,forward_gru/strided_slice_1/stack_1:output:0,forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
forward_gru/TensorArrayV2TensorListReserve0forward_gru/TensorArrayV2/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
3forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru/transpose:y:0Jforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
forward_gru/strided_slice_2StridedSliceforward_gru/transpose:y:0*forward_gru/strided_slice_2/stack:output:0,forward_gru/strided_slice_2/stack_1:output:0,forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
%forward_gru/gru_cell_1/ReadVariableOpReadVariableOp.forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0
forward_gru/gru_cell_1/unstackUnpack-forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¢
,forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
forward_gru/gru_cell_1/MatMulMatMul$forward_gru/strided_slice_2:output:04forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
forward_gru/gru_cell_1/BiasAddBiasAdd'forward_gru/gru_cell_1/MatMul:product:0'forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
forward_gru/gru_cell_1/splitSplit/forward_gru/gru_cell_1/split/split_dim:output:0'forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¦
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¯
forward_gru/gru_cell_1/MatMul_1MatMulforward_gru/zeros:output:06forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 forward_gru/gru_cell_1/BiasAdd_1BiasAdd)forward_gru/gru_cell_1/MatMul_1:product:0'forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿs
(forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
forward_gru/gru_cell_1/split_1SplitV)forward_gru/gru_cell_1/BiasAdd_1:output:0%forward_gru/gru_cell_1/Const:output:01forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¥
forward_gru/gru_cell_1/addAddV2%forward_gru/gru_cell_1/split:output:0'forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru/gru_cell_1/SigmoidSigmoidforward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
forward_gru/gru_cell_1/add_1AddV2%forward_gru/gru_cell_1/split:output:1'forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru/gru_cell_1/Sigmoid_1Sigmoid forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
forward_gru/gru_cell_1/mulMul$forward_gru/gru_cell_1/Sigmoid_1:y:0'forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_2AddV2%forward_gru/gru_cell_1/split:output:2forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
forward_gru/gru_cell_1/ReluRelu forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/mul_1Mul"forward_gru/gru_cell_1/Sigmoid:y:0forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
forward_gru/gru_cell_1/subSub%forward_gru/gru_cell_1/sub/x:output:0"forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
forward_gru/gru_cell_1/mul_2Mulforward_gru/gru_cell_1/sub:z:0)forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_3AddV2 forward_gru/gru_cell_1/mul_1:z:0 forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ü
forward_gru/TensorArrayV2_1TensorListReserve2forward_gru/TensorArrayV2_1/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
forward_gru/whileWhile'forward_gru/while/loop_counter:output:0-forward_gru/while/maximum_iterations:output:0forward_gru/time:output:0$forward_gru/TensorArrayV2_1:handle:0forward_gru/zeros:output:0$forward_gru/strided_slice_1:output:0Cforward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0.forward_gru_gru_cell_1_readvariableop_resource5forward_gru_gru_cell_1_matmul_readvariableop_resource7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *(
body R
forward_gru_while_body_18174*(
cond R
forward_gru_while_cond_18173*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
<forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   æ
.forward_gru/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru/while:output:3Eforward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0t
!forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
forward_gru/strided_slice_3StridedSlice7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0*forward_gru/strided_slice_3/stack:output:0,forward_gru/strided_slice_3/stack_1:output:0,forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskq
forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
forward_gru/transpose_1	Transpose7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0%forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    H
backward_gru/ShapeShapeinputs*
T0*
_output_shapes
:j
 backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_sliceStridedSlicebackward_gru/Shape:output:0)backward_gru/strided_slice/stack:output:0+backward_gru/strided_slice/stack_1:output:0+backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

backward_gru/zeros/packedPack#backward_gru/strided_slice:output:0$backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru/zerosFill"backward_gru/zeros/packed:output:0!backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru/transpose	Transposeinputs$backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
backward_gru/Shape_1Shapebackward_gru/transpose:y:0*
T0*
_output_shapes
:l
"backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_slice_1StridedSlicebackward_gru/Shape_1:output:0+backward_gru/strided_slice_1/stack:output:0-backward_gru/strided_slice_1/stack_1:output:0-backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
backward_gru/TensorArrayV2TensorListReserve1backward_gru/TensorArrayV2/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_gru/ReverseV2	ReverseV2backward_gru/transpose:y:0$backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
4backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbackward_gru/ReverseV2:output:0Kbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
backward_gru/strided_slice_2StridedSlicebackward_gru/transpose:y:0+backward_gru/strided_slice_2/stack:output:0-backward_gru/strided_slice_2/stack_1:output:0-backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
&backward_gru/gru_cell_2/ReadVariableOpReadVariableOp/backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0
backward_gru/gru_cell_2/unstackUnpack.backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¤
-backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
backward_gru/gru_cell_2/MatMulMatMul%backward_gru/strided_slice_2:output:05backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
backward_gru/gru_cell_2/BiasAddBiasAdd(backward_gru/gru_cell_2/MatMul:product:0(backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
backward_gru/gru_cell_2/splitSplit0backward_gru/gru_cell_2/split/split_dim:output:0(backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0²
 backward_gru/gru_cell_2/MatMul_1MatMulbackward_gru/zeros:output:07backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!backward_gru/gru_cell_2/BiasAdd_1BiasAdd*backward_gru/gru_cell_2/MatMul_1:product:0(backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿt
)backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
backward_gru/gru_cell_2/split_1SplitV*backward_gru/gru_cell_2/BiasAdd_1:output:0&backward_gru/gru_cell_2/Const:output:02backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
backward_gru/gru_cell_2/addAddV2&backward_gru/gru_cell_2/split:output:0(backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru/gru_cell_2/SigmoidSigmoidbackward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
backward_gru/gru_cell_2/add_1AddV2&backward_gru/gru_cell_2/split:output:1(backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru/gru_cell_2/Sigmoid_1Sigmoid!backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
backward_gru/gru_cell_2/mulMul%backward_gru/gru_cell_2/Sigmoid_1:y:0(backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
backward_gru/gru_cell_2/add_2AddV2&backward_gru/gru_cell_2/split:output:2backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
backward_gru/gru_cell_2/ReluRelu!backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/mul_1Mul#backward_gru/gru_cell_2/Sigmoid:y:0backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
backward_gru/gru_cell_2/subSub&backward_gru/gru_cell_2/sub/x:output:0#backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
backward_gru/gru_cell_2/mul_2Mulbackward_gru/gru_cell_2/sub:z:0*backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/add_3AddV2!backward_gru/gru_cell_2/mul_1:z:0!backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
*backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ß
backward_gru/TensorArrayV2_1TensorListReserve3backward_gru/TensorArrayV2_1/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
backward_gru/whileWhile(backward_gru/while/loop_counter:output:0.backward_gru/while/maximum_iterations:output:0backward_gru/time:output:0%backward_gru/TensorArrayV2_1:handle:0backward_gru/zeros:output:0%backward_gru/strided_slice_1:output:0Dbackward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/backward_gru_gru_cell_2_readvariableop_resource6backward_gru_gru_cell_2_matmul_readvariableop_resource8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *)
body!R
backward_gru_while_body_18325*)
cond!R
backward_gru_while_cond_18324*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
=backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   é
/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru/while:output:3Fbackward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
"backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
backward_gru/strided_slice_3StridedSlice8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0+backward_gru/strided_slice_3/stack:output:0-backward_gru/strided_slice_3/stack_1:output:0-backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskr
backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
backward_gru/transpose_1	Transpose8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0&backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2$forward_gru/strided_slice_3:output:0%backward_gru/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Å
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp.^backward_gru/gru_cell_2/MatMul/ReadVariableOp0^backward_gru/gru_cell_2/MatMul_1/ReadVariableOp'^backward_gru/gru_cell_2/ReadVariableOp^backward_gru/whileO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp-^forward_gru/gru_cell_1/MatMul/ReadVariableOp/^forward_gru/gru_cell_1/MatMul_1/ReadVariableOp&^forward_gru/gru_cell_1/ReadVariableOp^forward_gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_gru/gru_cell_2/MatMul/ReadVariableOp-backward_gru/gru_cell_2/MatMul/ReadVariableOp2b
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2P
&backward_gru/gru_cell_2/ReadVariableOp&backward_gru/gru_cell_2/ReadVariableOp2(
backward_gru/whilebackward_gru/while2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2\
,forward_gru/gru_cell_1/MatMul/ReadVariableOp,forward_gru/gru_cell_1/MatMul/ReadVariableOp2`
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2N
%forward_gru/gru_cell_1/ReadVariableOp%forward_gru/gru_cell_1/ReadVariableOp2&
forward_gru/whileforward_gru/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ê
__inference_loss_fn_1_22877i
Wbidirectional_backward_gru_gru_cell_2_kernel_regularizer_square_readvariableop_resource:
identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpæ
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpWbidirectional_backward_gru_gru_cell_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ~
IdentityIdentity@bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp
²¢
ð
!__inference__traced_restore_23144
file_prefix/
assignvariableop_dense_kernel:+
assignvariableop_1_dense_bias:3
!assignvariableop_2_dense_1_kernel:-
assignvariableop_3_dense_1_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: P
>assignvariableop_9_bidirectional_forward_gru_gru_cell_1_kernel:[
Iassignvariableop_10_bidirectional_forward_gru_gru_cell_1_recurrent_kernel:
O
=assignvariableop_11_bidirectional_forward_gru_gru_cell_1_bias:R
@assignvariableop_12_bidirectional_backward_gru_gru_cell_2_kernel:\
Jassignvariableop_13_bidirectional_backward_gru_gru_cell_2_recurrent_kernel:
P
>assignvariableop_14_bidirectional_backward_gru_gru_cell_2_bias:#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: 9
'assignvariableop_19_adam_dense_kernel_m:3
%assignvariableop_20_adam_dense_bias_m:;
)assignvariableop_21_adam_dense_1_kernel_m:5
'assignvariableop_22_adam_dense_1_bias_m:X
Fassignvariableop_23_adam_bidirectional_forward_gru_gru_cell_1_kernel_m:b
Passignvariableop_24_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_m:
V
Dassignvariableop_25_adam_bidirectional_forward_gru_gru_cell_1_bias_m:Y
Gassignvariableop_26_adam_bidirectional_backward_gru_gru_cell_2_kernel_m:c
Qassignvariableop_27_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_m:
W
Eassignvariableop_28_adam_bidirectional_backward_gru_gru_cell_2_bias_m:9
'assignvariableop_29_adam_dense_kernel_v:3
%assignvariableop_30_adam_dense_bias_v:;
)assignvariableop_31_adam_dense_1_kernel_v:5
'assignvariableop_32_adam_dense_1_bias_v:X
Fassignvariableop_33_adam_bidirectional_forward_gru_gru_cell_1_kernel_v:b
Passignvariableop_34_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_v:
V
Dassignvariableop_35_adam_bidirectional_forward_gru_gru_cell_1_bias_v:Y
Gassignvariableop_36_adam_bidirectional_backward_gru_gru_cell_2_kernel_v:c
Qassignvariableop_37_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_v:
W
Eassignvariableop_38_adam_bidirectional_backward_gru_gru_cell_2_bias_v:
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
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
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
:­
AssignVariableOp_9AssignVariableOp>assignvariableop_9_bidirectional_forward_gru_gru_cell_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_10AssignVariableOpIassignvariableop_10_bidirectional_forward_gru_gru_cell_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_11AssignVariableOp=assignvariableop_11_bidirectional_forward_gru_gru_cell_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_12AssignVariableOp@assignvariableop_12_bidirectional_backward_gru_gru_cell_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_13AssignVariableOpJassignvariableop_13_bidirectional_backward_gru_gru_cell_2_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_14AssignVariableOp>assignvariableop_14_bidirectional_backward_gru_gru_cell_2_biasIdentity_14:output:0"/device:CPU:0*
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
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_23AssignVariableOpFassignvariableop_23_adam_bidirectional_forward_gru_gru_cell_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_24AssignVariableOpPassignvariableop_24_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_25AssignVariableOpDassignvariableop_25_adam_bidirectional_forward_gru_gru_cell_1_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_26AssignVariableOpGassignvariableop_26_adam_bidirectional_backward_gru_gru_cell_2_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_27AssignVariableOpQassignvariableop_27_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOpEassignvariableop_28_adam_bidirectional_backward_gru_gru_cell_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_33AssignVariableOpFassignvariableop_33_adam_bidirectional_forward_gru_gru_cell_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_34AssignVariableOpPassignvariableop_34_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_35AssignVariableOpDassignvariableop_35_adam_bidirectional_forward_gru_gru_cell_1_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_36AssignVariableOpGassignvariableop_36_adam_bidirectional_backward_gru_gru_cell_2_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_37AssignVariableOpQassignvariableop_37_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_38AssignVariableOpEassignvariableop_38_adam_bidirectional_backward_gru_gru_cell_2_bias_vIdentity_38:output:0"/device:CPU:0*
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
³(
¤
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16668

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpf
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
¬
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
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
Ù
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
Õ
¥
while_cond_17389
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_17389___redundant_placeholder03
/while_while_cond_17389___redundant_placeholder13
/while_while_cond_17389___redundant_placeholder23
/while_while_cond_17389___redundant_placeholder3
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
Ä

E__inference_sequential_layer_call_and_return_conditional_losses_19808

inputsN
<bidirectional_forward_gru_gru_cell_1_readvariableop_resource:U
Cbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource:W
Ebidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
O
=bidirectional_backward_gru_gru_cell_2_readvariableop_resource:V
Dbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource:X
Fbidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource:
6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp¢=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢4bidirectional/backward_gru/gru_cell_2/ReadVariableOp¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ bidirectional/backward_gru/while¢:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp¢<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢3bidirectional/forward_gru/gru_cell_1/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢bidirectional/forward_gru/while¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOpU
bidirectional/forward_gru/ShapeShapeinputs*
T0*
_output_shapes
:w
-bidirectional/forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/bidirectional/forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/bidirectional/forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'bidirectional/forward_gru/strided_sliceStridedSlice(bidirectional/forward_gru/Shape:output:06bidirectional/forward_gru/strided_slice/stack:output:08bidirectional/forward_gru/strided_slice/stack_1:output:08bidirectional/forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(bidirectional/forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Á
&bidirectional/forward_gru/zeros/packedPack0bidirectional/forward_gru/strided_slice:output:01bidirectional/forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%bidirectional/forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
bidirectional/forward_gru/zerosFill/bidirectional/forward_gru/zeros/packed:output:0.bidirectional/forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
(bidirectional/forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¡
#bidirectional/forward_gru/transpose	Transposeinputs1bidirectional/forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
!bidirectional/forward_gru/Shape_1Shape'bidirectional/forward_gru/transpose:y:0*
T0*
_output_shapes
:y
/bidirectional/forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1bidirectional/forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)bidirectional/forward_gru/strided_slice_1StridedSlice*bidirectional/forward_gru/Shape_1:output:08bidirectional/forward_gru/strided_slice_1/stack:output:0:bidirectional/forward_gru/strided_slice_1/stack_1:output:0:bidirectional/forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5bidirectional/forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
'bidirectional/forward_gru/TensorArrayV2TensorListReserve>bidirectional/forward_gru/TensorArrayV2/element_shape:output:02bidirectional/forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ 
Obidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ®
Abidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'bidirectional/forward_gru/transpose:y:0Xbidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒy
/bidirectional/forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1bidirectional/forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
)bidirectional/forward_gru/strided_slice_2StridedSlice'bidirectional/forward_gru/transpose:y:08bidirectional/forward_gru/strided_slice_2/stack:output:0:bidirectional/forward_gru/strided_slice_2/stack_1:output:0:bidirectional/forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask°
3bidirectional/forward_gru/gru_cell_1/ReadVariableOpReadVariableOp<bidirectional_forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0©
,bidirectional/forward_gru/gru_cell_1/unstackUnpack;bidirectional/forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¾
:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOpCbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ß
+bidirectional/forward_gru/gru_cell_1/MatMulMatMul2bidirectional/forward_gru/strided_slice_2:output:0Bbidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
,bidirectional/forward_gru/gru_cell_1/BiasAddBiasAdd5bidirectional/forward_gru/gru_cell_1/MatMul:product:05bidirectional/forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4bidirectional/forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
*bidirectional/forward_gru/gru_cell_1/splitSplit=bidirectional/forward_gru/gru_cell_1/split/split_dim:output:05bidirectional/forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÂ
<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpEbidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0Ù
-bidirectional/forward_gru/gru_cell_1/MatMul_1MatMul(bidirectional/forward_gru/zeros:output:0Dbidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
.bidirectional/forward_gru/gru_cell_1/BiasAdd_1BiasAdd7bidirectional/forward_gru/gru_cell_1/MatMul_1:product:05bidirectional/forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*bidirectional/forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
6bidirectional/forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
,bidirectional/forward_gru/gru_cell_1/split_1SplitV7bidirectional/forward_gru/gru_cell_1/BiasAdd_1:output:03bidirectional/forward_gru/gru_cell_1/Const:output:0?bidirectional/forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÏ
(bidirectional/forward_gru/gru_cell_1/addAddV23bidirectional/forward_gru/gru_cell_1/split:output:05bidirectional/forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

,bidirectional/forward_gru/gru_cell_1/SigmoidSigmoid,bidirectional/forward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ñ
*bidirectional/forward_gru/gru_cell_1/add_1AddV23bidirectional/forward_gru/gru_cell_1/split:output:15bidirectional/forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

.bidirectional/forward_gru/gru_cell_1/Sigmoid_1Sigmoid.bidirectional/forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ì
(bidirectional/forward_gru/gru_cell_1/mulMul2bidirectional/forward_gru/gru_cell_1/Sigmoid_1:y:05bidirectional/forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È
*bidirectional/forward_gru/gru_cell_1/add_2AddV23bidirectional/forward_gru/gru_cell_1/split:output:2,bidirectional/forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)bidirectional/forward_gru/gru_cell_1/ReluRelu.bidirectional/forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
*bidirectional/forward_gru/gru_cell_1/mul_1Mul0bidirectional/forward_gru/gru_cell_1/Sigmoid:y:0(bidirectional/forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
*bidirectional/forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
(bidirectional/forward_gru/gru_cell_1/subSub3bidirectional/forward_gru/gru_cell_1/sub/x:output:00bidirectional/forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ê
*bidirectional/forward_gru/gru_cell_1/mul_2Mul,bidirectional/forward_gru/gru_cell_1/sub:z:07bidirectional/forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Å
*bidirectional/forward_gru/gru_cell_1/add_3AddV2.bidirectional/forward_gru/gru_cell_1/mul_1:z:0.bidirectional/forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

7bidirectional/forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
)bidirectional/forward_gru/TensorArrayV2_1TensorListReserve@bidirectional/forward_gru/TensorArrayV2_1/element_shape:output:02bidirectional/forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ`
bidirectional/forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2bidirectional/forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿn
,bidirectional/forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
bidirectional/forward_gru/whileWhile5bidirectional/forward_gru/while/loop_counter:output:0;bidirectional/forward_gru/while/maximum_iterations:output:0'bidirectional/forward_gru/time:output:02bidirectional/forward_gru/TensorArrayV2_1:handle:0(bidirectional/forward_gru/zeros:output:02bidirectional/forward_gru/strided_slice_1:output:0Qbidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0<bidirectional_forward_gru_gru_cell_1_readvariableop_resourceCbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resourceEbidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *6
body.R,
*bidirectional_forward_gru_while_body_19540*6
cond.R,
*bidirectional_forward_gru_while_cond_19539*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
Jbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
<bidirectional/forward_gru/TensorArrayV2Stack/TensorListStackTensorListStack(bidirectional/forward_gru/while:output:3Sbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
/bidirectional/forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1bidirectional/forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
)bidirectional/forward_gru/strided_slice_3StridedSliceEbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack:tensor:08bidirectional/forward_gru/strided_slice_3/stack:output:0:bidirectional/forward_gru/strided_slice_3/stack_1:output:0:bidirectional/forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
*bidirectional/forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ä
%bidirectional/forward_gru/transpose_1	TransposeEbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack:tensor:03bidirectional/forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
!bidirectional/forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
 bidirectional/backward_gru/ShapeShapeinputs*
T0*
_output_shapes
:x
.bidirectional/backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0bidirectional/backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0bidirectional/backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
(bidirectional/backward_gru/strided_sliceStridedSlice)bidirectional/backward_gru/Shape:output:07bidirectional/backward_gru/strided_slice/stack:output:09bidirectional/backward_gru/strided_slice/stack_1:output:09bidirectional/backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)bidirectional/backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Ä
'bidirectional/backward_gru/zeros/packedPack1bidirectional/backward_gru/strided_slice:output:02bidirectional/backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&bidirectional/backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
 bidirectional/backward_gru/zerosFill0bidirectional/backward_gru/zeros/packed:output:0/bidirectional/backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
)bidirectional/backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
$bidirectional/backward_gru/transpose	Transposeinputs2bidirectional/backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
"bidirectional/backward_gru/Shape_1Shape(bidirectional/backward_gru/transpose:y:0*
T0*
_output_shapes
:z
0bidirectional/backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*bidirectional/backward_gru/strided_slice_1StridedSlice+bidirectional/backward_gru/Shape_1:output:09bidirectional/backward_gru/strided_slice_1/stack:output:0;bidirectional/backward_gru/strided_slice_1/stack_1:output:0;bidirectional/backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6bidirectional/backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
(bidirectional/backward_gru/TensorArrayV2TensorListReserve?bidirectional/backward_gru/TensorArrayV2/element_shape:output:03bidirectional/backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)bidirectional/backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Å
$bidirectional/backward_gru/ReverseV2	ReverseV2(bidirectional/backward_gru/transpose:y:02bidirectional/backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
Pbidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¶
Bbidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor-bidirectional/backward_gru/ReverseV2:output:0Ybidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒz
0bidirectional/backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
*bidirectional/backward_gru/strided_slice_2StridedSlice(bidirectional/backward_gru/transpose:y:09bidirectional/backward_gru/strided_slice_2/stack:output:0;bidirectional/backward_gru/strided_slice_2/stack_1:output:0;bidirectional/backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask²
4bidirectional/backward_gru/gru_cell_2/ReadVariableOpReadVariableOp=bidirectional_backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0«
-bidirectional/backward_gru/gru_cell_2/unstackUnpack<bidirectional/backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÀ
;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOpDbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0â
,bidirectional/backward_gru/gru_cell_2/MatMulMatMul3bidirectional/backward_gru/strided_slice_2:output:0Cbidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
-bidirectional/backward_gru/gru_cell_2/BiasAddBiasAdd6bidirectional/backward_gru/gru_cell_2/MatMul:product:06bidirectional/backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5bidirectional/backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+bidirectional/backward_gru/gru_cell_2/splitSplit>bidirectional/backward_gru/gru_cell_2/split/split_dim:output:06bidirectional/backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÄ
=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0Ü
.bidirectional/backward_gru/gru_cell_2/MatMul_1MatMul)bidirectional/backward_gru/zeros:output:0Ebidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
/bidirectional/backward_gru/gru_cell_2/BiasAdd_1BiasAdd8bidirectional/backward_gru/gru_cell_2/MatMul_1:product:06bidirectional/backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+bidirectional/backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
7bidirectional/backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
-bidirectional/backward_gru/gru_cell_2/split_1SplitV8bidirectional/backward_gru/gru_cell_2/BiasAdd_1:output:04bidirectional/backward_gru/gru_cell_2/Const:output:0@bidirectional/backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÒ
)bidirectional/backward_gru/gru_cell_2/addAddV24bidirectional/backward_gru/gru_cell_2/split:output:06bidirectional/backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-bidirectional/backward_gru/gru_cell_2/SigmoidSigmoid-bidirectional/backward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô
+bidirectional/backward_gru/gru_cell_2/add_1AddV24bidirectional/backward_gru/gru_cell_2/split:output:16bidirectional/backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/bidirectional/backward_gru/gru_cell_2/Sigmoid_1Sigmoid/bidirectional/backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ï
)bidirectional/backward_gru/gru_cell_2/mulMul3bidirectional/backward_gru/gru_cell_2/Sigmoid_1:y:06bidirectional/backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ë
+bidirectional/backward_gru/gru_cell_2/add_2AddV24bidirectional/backward_gru/gru_cell_2/split:output:2-bidirectional/backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*bidirectional/backward_gru/gru_cell_2/ReluRelu/bidirectional/backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
+bidirectional/backward_gru/gru_cell_2/mul_1Mul1bidirectional/backward_gru/gru_cell_2/Sigmoid:y:0)bidirectional/backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
+bidirectional/backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ë
)bidirectional/backward_gru/gru_cell_2/subSub4bidirectional/backward_gru/gru_cell_2/sub/x:output:01bidirectional/backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Í
+bidirectional/backward_gru/gru_cell_2/mul_2Mul-bidirectional/backward_gru/gru_cell_2/sub:z:08bidirectional/backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È
+bidirectional/backward_gru/gru_cell_2/add_3AddV2/bidirectional/backward_gru/gru_cell_2/mul_1:z:0/bidirectional/backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

8bidirectional/backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
*bidirectional/backward_gru/TensorArrayV2_1TensorListReserveAbidirectional/backward_gru/TensorArrayV2_1/element_shape:output:03bidirectional/backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒa
bidirectional/backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3bidirectional/backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿo
-bidirectional/backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
 bidirectional/backward_gru/whileWhile6bidirectional/backward_gru/while/loop_counter:output:0<bidirectional/backward_gru/while/maximum_iterations:output:0(bidirectional/backward_gru/time:output:03bidirectional/backward_gru/TensorArrayV2_1:handle:0)bidirectional/backward_gru/zeros:output:03bidirectional/backward_gru/strided_slice_1:output:0Rbidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=bidirectional_backward_gru_gru_cell_2_readvariableop_resourceDbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resourceFbidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *7
body/R-
+bidirectional_backward_gru_while_body_19691*7
cond/R-
+bidirectional_backward_gru_while_cond_19690*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
Kbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
=bidirectional/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStack)bidirectional/backward_gru/while:output:3Tbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
0bidirectional/backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ|
2bidirectional/backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
*bidirectional/backward_gru/strided_slice_3StridedSliceFbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack:tensor:09bidirectional/backward_gru/strided_slice_3/stack:output:0;bidirectional/backward_gru/strided_slice_3/stack_1:output:0;bidirectional/backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
+bidirectional/backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ç
&bidirectional/backward_gru/transpose_1	TransposeFbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack:tensor:04bidirectional/backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
"bidirectional/backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    [
bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :è
bidirectional/concatConcatV22bidirectional/forward_gru/strided_slice_3:output:03bidirectional/backward_gru/strided_slice_3:output:0"bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMulbidirectional/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ó
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp<^bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp>^bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp5^bidirectional/backward_gru/gru_cell_2/ReadVariableOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^bidirectional/backward_gru/while;^bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp=^bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp4^bidirectional/forward_gru/gru_cell_1/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp ^bidirectional/forward_gru/while^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2z
;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp2~
=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2l
4bidirectional/backward_gru/gru_cell_2/ReadVariableOp4bidirectional/backward_gru/gru_cell_2/ReadVariableOp2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 bidirectional/backward_gru/while bidirectional/backward_gru/while2x
:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp2|
<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2j
3bidirectional/forward_gru/gru_cell_1/ReadVariableOp3bidirectional/forward_gru/gru_cell_1/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2B
bidirectional/forward_gru/whilebidirectional/forward_gru/while2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_21340
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21340___redundant_placeholder03
/while_while_cond_21340___redundant_placeholder13
/while_while_cond_21340___redundant_placeholder23
/while_while_cond_21340___redundant_placeholder3
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
Â(
¥
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17187

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpf
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
­
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
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
Ú
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
ÁK
à

forward_gru_while_body_181744
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_23
/forward_gru_while_forward_gru_strided_slice_1_0o
kforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0H
6forward_gru_while_gru_cell_1_readvariableop_resource_0:O
=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:Q
?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:

forward_gru_while_identity 
forward_gru_while_identity_1 
forward_gru_while_identity_2 
forward_gru_while_identity_3 
forward_gru_while_identity_41
-forward_gru_while_forward_gru_strided_slice_1m
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorF
4forward_gru_while_gru_cell_1_readvariableop_resource:M
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource:O
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢+forward_gru/while/gru_cell_1/ReadVariableOp
Cforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   â
5forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0forward_gru_while_placeholderLforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¢
+forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOp6forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
$forward_gru/while/gru_cell_1/unstackUnpack3forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num°
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
#forward_gru/while/gru_cell_1/MatMulMatMul<forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0:forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$forward_gru/while/gru_cell_1/BiasAddBiasAdd-forward_gru/while/gru_cell_1/MatMul:product:0-forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
"forward_gru/while/gru_cell_1/splitSplit5forward_gru/while/gru_cell_1/split/split_dim:output:0-forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split´
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0À
%forward_gru/while/gru_cell_1/MatMul_1MatMulforward_gru_while_placeholder_2<forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
&forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd/forward_gru/while/gru_cell_1/MatMul_1:product:0-forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿy
.forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
$forward_gru/while/gru_cell_1/split_1SplitV/forward_gru/while/gru_cell_1/BiasAdd_1:output:0+forward_gru/while/gru_cell_1/Const:output:07forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split·
 forward_gru/while/gru_cell_1/addAddV2+forward_gru/while/gru_cell_1/split:output:0-forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$forward_gru/while/gru_cell_1/SigmoidSigmoid$forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
"forward_gru/while/gru_cell_1/add_1AddV2+forward_gru/while/gru_cell_1/split:output:1-forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid&forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
 forward_gru/while/gru_cell_1/mulMul*forward_gru/while/gru_cell_1/Sigmoid_1:y:0-forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
"forward_gru/while/gru_cell_1/add_2AddV2+forward_gru/while/gru_cell_1/split:output:2$forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!forward_gru/while/gru_cell_1/ReluRelu&forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
"forward_gru/while/gru_cell_1/mul_1Mul(forward_gru/while/gru_cell_1/Sigmoid:y:0forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
"forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
 forward_gru/while/gru_cell_1/subSub+forward_gru/while/gru_cell_1/sub/x:output:0(forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
"forward_gru/while/gru_cell_1/mul_2Mul$forward_gru/while/gru_cell_1/sub:z:0/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
"forward_gru/while/gru_cell_1/add_3AddV2&forward_gru/while/gru_cell_1/mul_1:z:0&forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
6forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemforward_gru_while_placeholder_1forward_gru_while_placeholder&forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒY
forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/addAddV2forward_gru_while_placeholder forward_gru/while/add/y:output:0*
T0*
_output_shapes
: [
forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/add_1AddV20forward_gru_while_forward_gru_while_loop_counter"forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: }
forward_gru/while/IdentityIdentityforward_gru/while/add_1:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: 
forward_gru/while/Identity_1Identity6forward_gru_while_forward_gru_while_maximum_iterations^forward_gru/while/NoOp*
T0*
_output_shapes
: }
forward_gru/while/Identity_2Identityforward_gru/while/add:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: ½
forward_gru/while/Identity_3IdentityFforward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
forward_gru/while/Identity_4Identity&forward_gru/while/gru_cell_1/add_3:z:0^forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
forward_gru/while/NoOpNoOp3^forward_gru/while/gru_cell_1/MatMul/ReadVariableOp5^forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp,^forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "`
-forward_gru_while_forward_gru_strided_slice_1/forward_gru_while_forward_gru_strided_slice_1_0"
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"|
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"n
4forward_gru_while_gru_cell_1_readvariableop_resource6forward_gru_while_gru_cell_1_readvariableop_resource_0"A
forward_gru_while_identity#forward_gru/while/Identity:output:0"E
forward_gru_while_identity_1%forward_gru/while/Identity_1:output:0"E
forward_gru_while_identity_2%forward_gru/while/Identity_2:output:0"E
forward_gru_while_identity_3%forward_gru/while/Identity_3:output:0"E
forward_gru_while_identity_4%forward_gru/while/Identity_4:output:0"Ø
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2h
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2l
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2Z
+forward_gru/while/gru_cell_1/ReadVariableOp+forward_gru/while/gru_cell_1/ReadVariableOp: 
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
¢
¹
+__inference_forward_gru_layer_call_fn_21266

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallè
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
GPU 2J 8 *O
fJRH
F__inference_forward_gru_layer_call_and_return_conditional_losses_17485o
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
ÕX
Ó
F__inference_forward_gru_layer_call_and_return_conditional_losses_21913

inputs4
"gru_cell_1_readvariableop_resource:;
)gru_cell_1_matmul_readvariableop_resource:=
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
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
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
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
while_body_21818*
condR
while_cond_21817*8
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
 *    ·
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹<
ø
while_body_22190
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:C
1while_gru_cell_2_matmul_readvariableop_resource_0:E
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:A
/while_gru_cell_2_matmul_readvariableop_resource:C
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
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
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp: 
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
Ù
÷
6sequential_bidirectional_backward_gru_while_cond_16486h
dsequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_loop_countern
jsequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_maximum_iterations;
7sequential_bidirectional_backward_gru_while_placeholder=
9sequential_bidirectional_backward_gru_while_placeholder_1=
9sequential_bidirectional_backward_gru_while_placeholder_2j
fsequential_bidirectional_backward_gru_while_less_sequential_bidirectional_backward_gru_strided_slice_1
{sequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_cond_16486___redundant_placeholder0
{sequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_cond_16486___redundant_placeholder1
{sequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_cond_16486___redundant_placeholder2
{sequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_cond_16486___redundant_placeholder38
4sequential_bidirectional_backward_gru_while_identity
ú
0sequential/bidirectional/backward_gru/while/LessLess7sequential_bidirectional_backward_gru_while_placeholderfsequential_bidirectional_backward_gru_while_less_sequential_bidirectional_backward_gru_strided_slice_1*
T0*
_output_shapes
: 
4sequential/bidirectional/backward_gru/while/IdentityIdentity4sequential/bidirectional/backward_gru/while/Less:z:0*
T0
*
_output_shapes
: "u
4sequential_bidirectional_backward_gru_while_identity=sequential/bidirectional/backward_gru/while/Identity:output:0*(
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
¨

û
*__inference_sequential_layer_call_fn_19144

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
identity¢StatefulPartitionedCallÂ
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
GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18950o
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
ÊK
à

forward_gru_while_body_199794
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_23
/forward_gru_while_forward_gru_strided_slice_1_0o
kforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0H
6forward_gru_while_gru_cell_1_readvariableop_resource_0:O
=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:Q
?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:

forward_gru_while_identity 
forward_gru_while_identity_1 
forward_gru_while_identity_2 
forward_gru_while_identity_3 
forward_gru_while_identity_41
-forward_gru_while_forward_gru_strided_slice_1m
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorF
4forward_gru_while_gru_cell_1_readvariableop_resource:M
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource:O
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢+forward_gru/while/gru_cell_1/ReadVariableOp
Cforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿë
5forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0forward_gru_while_placeholderLforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¢
+forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOp6forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
$forward_gru/while/gru_cell_1/unstackUnpack3forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num°
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
#forward_gru/while/gru_cell_1/MatMulMatMul<forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0:forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$forward_gru/while/gru_cell_1/BiasAddBiasAdd-forward_gru/while/gru_cell_1/MatMul:product:0-forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
"forward_gru/while/gru_cell_1/splitSplit5forward_gru/while/gru_cell_1/split/split_dim:output:0-forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split´
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0À
%forward_gru/while/gru_cell_1/MatMul_1MatMulforward_gru_while_placeholder_2<forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
&forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd/forward_gru/while/gru_cell_1/MatMul_1:product:0-forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿy
.forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
$forward_gru/while/gru_cell_1/split_1SplitV/forward_gru/while/gru_cell_1/BiasAdd_1:output:0+forward_gru/while/gru_cell_1/Const:output:07forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split·
 forward_gru/while/gru_cell_1/addAddV2+forward_gru/while/gru_cell_1/split:output:0-forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$forward_gru/while/gru_cell_1/SigmoidSigmoid$forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
"forward_gru/while/gru_cell_1/add_1AddV2+forward_gru/while/gru_cell_1/split:output:1-forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid&forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
 forward_gru/while/gru_cell_1/mulMul*forward_gru/while/gru_cell_1/Sigmoid_1:y:0-forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
"forward_gru/while/gru_cell_1/add_2AddV2+forward_gru/while/gru_cell_1/split:output:2$forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!forward_gru/while/gru_cell_1/ReluRelu&forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
"forward_gru/while/gru_cell_1/mul_1Mul(forward_gru/while/gru_cell_1/Sigmoid:y:0forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
"forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
 forward_gru/while/gru_cell_1/subSub+forward_gru/while/gru_cell_1/sub/x:output:0(forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
"forward_gru/while/gru_cell_1/mul_2Mul$forward_gru/while/gru_cell_1/sub:z:0/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
"forward_gru/while/gru_cell_1/add_3AddV2&forward_gru/while/gru_cell_1/mul_1:z:0&forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
6forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemforward_gru_while_placeholder_1forward_gru_while_placeholder&forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒY
forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/addAddV2forward_gru_while_placeholder forward_gru/while/add/y:output:0*
T0*
_output_shapes
: [
forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/add_1AddV20forward_gru_while_forward_gru_while_loop_counter"forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: }
forward_gru/while/IdentityIdentityforward_gru/while/add_1:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: 
forward_gru/while/Identity_1Identity6forward_gru_while_forward_gru_while_maximum_iterations^forward_gru/while/NoOp*
T0*
_output_shapes
: }
forward_gru/while/Identity_2Identityforward_gru/while/add:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: ½
forward_gru/while/Identity_3IdentityFforward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
forward_gru/while/Identity_4Identity&forward_gru/while/gru_cell_1/add_3:z:0^forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
forward_gru/while/NoOpNoOp3^forward_gru/while/gru_cell_1/MatMul/ReadVariableOp5^forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp,^forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "`
-forward_gru_while_forward_gru_strided_slice_1/forward_gru_while_forward_gru_strided_slice_1_0"
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"|
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"n
4forward_gru_while_gru_cell_1_readvariableop_resource6forward_gru_while_gru_cell_1_readvariableop_resource_0"A
forward_gru_while_identity#forward_gru/while/Identity:output:0"E
forward_gru_while_identity_1%forward_gru/while/Identity_1:output:0"E
forward_gru_while_identity_2%forward_gru/while/Identity_2:output:0"E
forward_gru_while_identity_3%forward_gru/while/Identity_3:output:0"E
forward_gru_while_identity_4%forward_gru/while/Identity_4:output:0"Ø
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2h
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2l
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2Z
+forward_gru/while/gru_cell_1/ReadVariableOp+forward_gru/while/gru_cell_1/ReadVariableOp: 
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
while_body_21818
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_1_readvariableop_resource_0:C
1while_gru_cell_1_matmul_readvariableop_resource_0:E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_1_readvariableop_resource:A
/while_gru_cell_1_matmul_readvariableop_resource:C
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
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
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 
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
¿'
¶
H__inference_bidirectional_layer_call_and_return_conditional_losses_17676

inputs#
forward_gru_17486:#
forward_gru_17488:#
forward_gru_17490:
$
backward_gru_17654:$
backward_gru_17656:$
backward_gru_17658:

identity¢$backward_gru/StatefulPartitionedCall¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢#forward_gru/StatefulPartitionedCall
#forward_gru/StatefulPartitionedCallStatefulPartitionedCallinputsforward_gru_17486forward_gru_17488forward_gru_17490*
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
GPU 2J 8 *O
fJRH
F__inference_forward_gru_layer_call_and_return_conditional_losses_17485
$backward_gru/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_gru_17654backward_gru_17656backward_gru_17658*
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
GPU 2J 8 *P
fKRI
G__inference_backward_gru_layer_call_and_return_conditional_losses_17653M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :À
concatConcatV2,forward_gru/StatefulPartitionedCall:output:0-backward_gru/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpforward_gru_17488*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¡
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbackward_gru_17656*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
NoOpNoOp%^backward_gru/StatefulPartitionedCallO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp$^forward_gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2L
$backward_gru/StatefulPartitionedCall$backward_gru/StatefulPartitionedCall2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2J
#forward_gru/StatefulPartitionedCall#forward_gru/StatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

%__inference_dense_layer_call_fn_21196

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÕ
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
GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18453o
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

»
+__inference_forward_gru_layer_call_fn_21255
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallê
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
GPU 2J 8 *O
fJRH
F__inference_forward_gru_layer_call_and_return_conditional_losses_16945o
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
¨

û
*__inference_sequential_layer_call_fn_19119

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
identity¢StatefulPartitionedCallÂ
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
GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18489o
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
+
Þ
E__inference_sequential_layer_call_and_return_conditional_losses_19037
bidirectional_input%
bidirectional_19001:%
bidirectional_19003:%
bidirectional_19005:
%
bidirectional_19007:%
bidirectional_19009:%
bidirectional_19011:

dense_19014:
dense_19016:
dense_1_19019:
dense_1_19021:
identity¢%bidirectional/StatefulPartitionedCall¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallê
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputbidirectional_19001bidirectional_19003bidirectional_19005bidirectional_19007bidirectional_19009bidirectional_19011*
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18428
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_19014dense_19016*
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
GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18453
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_19019dense_1_19021*
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
B__inference_dense_1_layer_call_and_return_conditional_losses_18470¡
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_19003*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_19009*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp&^bidirectional/StatefulPartitionedCallO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namebidirectional_input
¹<
ø
while_body_21500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_1_readvariableop_resource_0:C
1while_gru_cell_1_matmul_readvariableop_resource_0:E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_1_readvariableop_resource:A
/while_gru_cell_1_matmul_readvariableop_resource:C
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
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
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 
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
while_body_17241
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_2_17263_0:*
while_gru_cell_2_17265_0:*
while_gru_cell_2_17267_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_2_17263:(
while_gru_cell_2_17265:(
while_gru_cell_2_17267:
¢(while/gru_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_17263_0while_gru_cell_2_17265_0while_gru_cell_2_17267_0*
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17187Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_2_17263while_gru_cell_2_17263_0"2
while_gru_cell_2_17265while_gru_cell_2_17265_0"2
while_gru_cell_2_17267while_gru_cell_2_17267_0")
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
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall: 
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
Â(
¥
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17032

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpf
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
­
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
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
Ú
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates
­

Ö
*__inference_gru_cell_2_layer_call_fn_22762

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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17032o
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

»
+__inference_forward_gru_layer_call_fn_21244
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallê
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
GPU 2J 8 *O
fJRH
F__inference_forward_gru_layer_call_and_return_conditional_losses_16751o
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
]

*bidirectional_forward_gru_while_body_19208P
Lbidirectional_forward_gru_while_bidirectional_forward_gru_while_loop_counterV
Rbidirectional_forward_gru_while_bidirectional_forward_gru_while_maximum_iterations/
+bidirectional_forward_gru_while_placeholder1
-bidirectional_forward_gru_while_placeholder_11
-bidirectional_forward_gru_while_placeholder_2O
Kbidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1_0
bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0V
Dbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0:]
Kbidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:_
Mbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:
,
(bidirectional_forward_gru_while_identity.
*bidirectional_forward_gru_while_identity_1.
*bidirectional_forward_gru_while_identity_2.
*bidirectional_forward_gru_while_identity_3.
*bidirectional_forward_gru_while_identity_4M
Ibidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1
bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensorT
Bbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource:[
Ibidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource:]
Kbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢Bbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp¢
Qbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ©
Cbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0+bidirectional_forward_gru_while_placeholderZbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¾
9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOpDbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
2bidirectional/forward_gru/while/gru_cell_1/unstackUnpackAbidirectional/forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÌ
@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOpKbidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
1bidirectional/forward_gru/while/gru_cell_1/MatMulMatMulJbidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0Hbidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
2bidirectional/forward_gru/while/gru_cell_1/BiasAddBiasAdd;bidirectional/forward_gru/while/gru_cell_1/MatMul:product:0;bidirectional/forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:bidirectional/forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
0bidirectional/forward_gru/while/gru_cell_1/splitSplitCbidirectional/forward_gru/while/gru_cell_1/split/split_dim:output:0;bidirectional/forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÐ
Bbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpMbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0ê
3bidirectional/forward_gru/while/gru_cell_1/MatMul_1MatMul-bidirectional_forward_gru_while_placeholder_2Jbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
4bidirectional/forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd=bidirectional/forward_gru/while/gru_cell_1/MatMul_1:product:0;bidirectional/forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0bidirectional/forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
<bidirectional/forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
2bidirectional/forward_gru/while/gru_cell_1/split_1SplitV=bidirectional/forward_gru/while/gru_cell_1/BiasAdd_1:output:09bidirectional/forward_gru/while/gru_cell_1/Const:output:0Ebidirectional/forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitá
.bidirectional/forward_gru/while/gru_cell_1/addAddV29bidirectional/forward_gru/while/gru_cell_1/split:output:0;bidirectional/forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
2bidirectional/forward_gru/while/gru_cell_1/SigmoidSigmoid2bidirectional/forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ã
0bidirectional/forward_gru/while/gru_cell_1/add_1AddV29bidirectional/forward_gru/while/gru_cell_1/split:output:1;bidirectional/forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
4bidirectional/forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid4bidirectional/forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Þ
.bidirectional/forward_gru/while/gru_cell_1/mulMul8bidirectional/forward_gru/while/gru_cell_1/Sigmoid_1:y:0;bidirectional/forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
0bidirectional/forward_gru/while/gru_cell_1/add_2AddV29bidirectional/forward_gru/while/gru_cell_1/split:output:22bidirectional/forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/bidirectional/forward_gru/while/gru_cell_1/ReluRelu4bidirectional/forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ð
0bidirectional/forward_gru/while/gru_cell_1/mul_1Mul6bidirectional/forward_gru/while/gru_cell_1/Sigmoid:y:0-bidirectional_forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
0bidirectional/forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ú
.bidirectional/forward_gru/while/gru_cell_1/subSub9bidirectional/forward_gru/while/gru_cell_1/sub/x:output:06bidirectional/forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ü
0bidirectional/forward_gru/while/gru_cell_1/mul_2Mul2bidirectional/forward_gru/while/gru_cell_1/sub:z:0=bidirectional/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
0bidirectional/forward_gru/while/gru_cell_1/add_3AddV24bidirectional/forward_gru/while/gru_cell_1/mul_1:z:04bidirectional/forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
Dbidirectional/forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-bidirectional_forward_gru_while_placeholder_1+bidirectional_forward_gru_while_placeholder4bidirectional/forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒg
%bidirectional/forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ª
#bidirectional/forward_gru/while/addAddV2+bidirectional_forward_gru_while_placeholder.bidirectional/forward_gru/while/add/y:output:0*
T0*
_output_shapes
: i
'bidirectional/forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ï
%bidirectional/forward_gru/while/add_1AddV2Lbidirectional_forward_gru_while_bidirectional_forward_gru_while_loop_counter0bidirectional/forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: §
(bidirectional/forward_gru/while/IdentityIdentity)bidirectional/forward_gru/while/add_1:z:0%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: Ò
*bidirectional/forward_gru/while/Identity_1IdentityRbidirectional_forward_gru_while_bidirectional_forward_gru_while_maximum_iterations%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: §
*bidirectional/forward_gru/while/Identity_2Identity'bidirectional/forward_gru/while/add:z:0%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: ç
*bidirectional/forward_gru/while/Identity_3IdentityTbidirectional/forward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒÅ
*bidirectional/forward_gru/while/Identity_4Identity4bidirectional/forward_gru/while/gru_cell_1/add_3:z:0%^bidirectional/forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
$bidirectional/forward_gru/while/NoOpNoOpA^bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpC^bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:^bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Ibidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1Kbidirectional_forward_gru_while_bidirectional_forward_gru_strided_slice_1_0"
Kbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resourceMbidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"
Ibidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resourceKbidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"
Bbidirectional_forward_gru_while_gru_cell_1_readvariableop_resourceDbidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0"]
(bidirectional_forward_gru_while_identity1bidirectional/forward_gru/while/Identity:output:0"a
*bidirectional_forward_gru_while_identity_13bidirectional/forward_gru/while/Identity_1:output:0"a
*bidirectional_forward_gru_while_identity_23bidirectional/forward_gru/while/Identity_2:output:0"a
*bidirectional_forward_gru_while_identity_33bidirectional/forward_gru/while/Identity_3:output:0"a
*bidirectional_forward_gru_while_identity_43bidirectional/forward_gru/while/Identity_4:output:0"
bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensorbidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp@bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2
Bbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpBbidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2v
9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp9bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp: 
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
while_cond_17044
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_17044___redundant_placeholder03
/while_while_cond_17044___redundant_placeholder13
/while_while_cond_17044___redundant_placeholder23
/while_while_cond_17044___redundant_placeholder3
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
ÕX
Ó
F__inference_forward_gru_layer_call_and_return_conditional_losses_21754

inputs4
"gru_cell_1_readvariableop_resource:;
)gru_cell_1_matmul_readvariableop_resource:=
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
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
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
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
while_body_21659*
condR
while_cond_21658*8
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
 *    ·
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹W
 
__inference__traced_save_23017
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_bidirectional_forward_gru_gru_cell_1_kernel_read_readvariableopT
Psavev2_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_read_readvariableopH
Dsavev2_bidirectional_forward_gru_gru_cell_1_bias_read_readvariableopK
Gsavev2_bidirectional_backward_gru_gru_cell_2_kernel_read_readvariableopU
Qsavev2_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_read_readvariableopI
Esavev2_bidirectional_backward_gru_gru_cell_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopQ
Msavev2_adam_bidirectional_forward_gru_gru_cell_1_kernel_m_read_readvariableop[
Wsavev2_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_m_read_readvariableopO
Ksavev2_adam_bidirectional_forward_gru_gru_cell_1_bias_m_read_readvariableopR
Nsavev2_adam_bidirectional_backward_gru_gru_cell_2_kernel_m_read_readvariableop\
Xsavev2_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_m_read_readvariableopP
Lsavev2_adam_bidirectional_backward_gru_gru_cell_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopQ
Msavev2_adam_bidirectional_forward_gru_gru_cell_1_kernel_v_read_readvariableop[
Wsavev2_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_v_read_readvariableopO
Ksavev2_adam_bidirectional_forward_gru_gru_cell_1_bias_v_read_readvariableopR
Nsavev2_adam_bidirectional_backward_gru_gru_cell_2_kernel_v_read_readvariableop\
Xsavev2_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_v_read_readvariableopP
Lsavev2_adam_bidirectional_backward_gru_gru_cell_2_bias_v_read_readvariableop
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
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ð
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_bidirectional_forward_gru_gru_cell_1_kernel_read_readvariableopPsavev2_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_read_readvariableopDsavev2_bidirectional_forward_gru_gru_cell_1_bias_read_readvariableopGsavev2_bidirectional_backward_gru_gru_cell_2_kernel_read_readvariableopQsavev2_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_read_readvariableopEsavev2_bidirectional_backward_gru_gru_cell_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopMsavev2_adam_bidirectional_forward_gru_gru_cell_1_kernel_m_read_readvariableopWsavev2_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_m_read_readvariableopKsavev2_adam_bidirectional_forward_gru_gru_cell_1_bias_m_read_readvariableopNsavev2_adam_bidirectional_backward_gru_gru_cell_2_kernel_m_read_readvariableopXsavev2_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_m_read_readvariableopLsavev2_adam_bidirectional_backward_gru_gru_cell_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopMsavev2_adam_bidirectional_forward_gru_gru_cell_1_kernel_v_read_readvariableopWsavev2_adam_bidirectional_forward_gru_gru_cell_1_recurrent_kernel_v_read_readvariableopKsavev2_adam_bidirectional_forward_gru_gru_cell_1_bias_v_read_readvariableopNsavev2_adam_bidirectional_backward_gru_gru_cell_2_kernel_v_read_readvariableopXsavev2_adam_bidirectional_backward_gru_gru_cell_2_recurrent_kernel_v_read_readvariableopLsavev2_adam_bidirectional_backward_gru_gru_cell_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
­

Ö
*__inference_gru_cell_1_layer_call_fn_22627

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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16668o
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
µ


backward_gru_while_cond_210836
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_28
4backward_gru_while_less_backward_gru_strided_slice_1M
Ibackward_gru_while_backward_gru_while_cond_21083___redundant_placeholder0M
Ibackward_gru_while_backward_gru_while_cond_21083___redundant_placeholder1M
Ibackward_gru_while_backward_gru_while_cond_21083___redundant_placeholder2M
Ibackward_gru_while_backward_gru_while_cond_21083___redundant_placeholder3
backward_gru_while_identity

backward_gru/while/LessLessbackward_gru_while_placeholder4backward_gru_while_less_backward_gru_strided_slice_1*
T0*
_output_shapes
: e
backward_gru/while/IdentityIdentitybackward_gru/while/Less:z:0*
T0
*
_output_shapes
: "C
backward_gru_while_identity$backward_gru/while/Identity:output:0*(
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
Ä

E__inference_sequential_layer_call_and_return_conditional_losses_19476

inputsN
<bidirectional_forward_gru_gru_cell_1_readvariableop_resource:U
Cbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource:W
Ebidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
O
=bidirectional_backward_gru_gru_cell_2_readvariableop_resource:V
Dbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource:X
Fbidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource:
6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:
identity¢;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp¢=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢4bidirectional/backward_gru/gru_cell_2/ReadVariableOp¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ bidirectional/backward_gru/while¢:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp¢<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢3bidirectional/forward_gru/gru_cell_1/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢bidirectional/forward_gru/while¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOpU
bidirectional/forward_gru/ShapeShapeinputs*
T0*
_output_shapes
:w
-bidirectional/forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/bidirectional/forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/bidirectional/forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'bidirectional/forward_gru/strided_sliceStridedSlice(bidirectional/forward_gru/Shape:output:06bidirectional/forward_gru/strided_slice/stack:output:08bidirectional/forward_gru/strided_slice/stack_1:output:08bidirectional/forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(bidirectional/forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Á
&bidirectional/forward_gru/zeros/packedPack0bidirectional/forward_gru/strided_slice:output:01bidirectional/forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%bidirectional/forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
bidirectional/forward_gru/zerosFill/bidirectional/forward_gru/zeros/packed:output:0.bidirectional/forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
(bidirectional/forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¡
#bidirectional/forward_gru/transpose	Transposeinputs1bidirectional/forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
!bidirectional/forward_gru/Shape_1Shape'bidirectional/forward_gru/transpose:y:0*
T0*
_output_shapes
:y
/bidirectional/forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1bidirectional/forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)bidirectional/forward_gru/strided_slice_1StridedSlice*bidirectional/forward_gru/Shape_1:output:08bidirectional/forward_gru/strided_slice_1/stack:output:0:bidirectional/forward_gru/strided_slice_1/stack_1:output:0:bidirectional/forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5bidirectional/forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
'bidirectional/forward_gru/TensorArrayV2TensorListReserve>bidirectional/forward_gru/TensorArrayV2/element_shape:output:02bidirectional/forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ 
Obidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ®
Abidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'bidirectional/forward_gru/transpose:y:0Xbidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒy
/bidirectional/forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1bidirectional/forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
)bidirectional/forward_gru/strided_slice_2StridedSlice'bidirectional/forward_gru/transpose:y:08bidirectional/forward_gru/strided_slice_2/stack:output:0:bidirectional/forward_gru/strided_slice_2/stack_1:output:0:bidirectional/forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask°
3bidirectional/forward_gru/gru_cell_1/ReadVariableOpReadVariableOp<bidirectional_forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0©
,bidirectional/forward_gru/gru_cell_1/unstackUnpack;bidirectional/forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¾
:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOpCbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ß
+bidirectional/forward_gru/gru_cell_1/MatMulMatMul2bidirectional/forward_gru/strided_slice_2:output:0Bbidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
,bidirectional/forward_gru/gru_cell_1/BiasAddBiasAdd5bidirectional/forward_gru/gru_cell_1/MatMul:product:05bidirectional/forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4bidirectional/forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
*bidirectional/forward_gru/gru_cell_1/splitSplit=bidirectional/forward_gru/gru_cell_1/split/split_dim:output:05bidirectional/forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÂ
<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpEbidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0Ù
-bidirectional/forward_gru/gru_cell_1/MatMul_1MatMul(bidirectional/forward_gru/zeros:output:0Dbidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
.bidirectional/forward_gru/gru_cell_1/BiasAdd_1BiasAdd7bidirectional/forward_gru/gru_cell_1/MatMul_1:product:05bidirectional/forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*bidirectional/forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
6bidirectional/forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
,bidirectional/forward_gru/gru_cell_1/split_1SplitV7bidirectional/forward_gru/gru_cell_1/BiasAdd_1:output:03bidirectional/forward_gru/gru_cell_1/Const:output:0?bidirectional/forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÏ
(bidirectional/forward_gru/gru_cell_1/addAddV23bidirectional/forward_gru/gru_cell_1/split:output:05bidirectional/forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

,bidirectional/forward_gru/gru_cell_1/SigmoidSigmoid,bidirectional/forward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ñ
*bidirectional/forward_gru/gru_cell_1/add_1AddV23bidirectional/forward_gru/gru_cell_1/split:output:15bidirectional/forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

.bidirectional/forward_gru/gru_cell_1/Sigmoid_1Sigmoid.bidirectional/forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ì
(bidirectional/forward_gru/gru_cell_1/mulMul2bidirectional/forward_gru/gru_cell_1/Sigmoid_1:y:05bidirectional/forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È
*bidirectional/forward_gru/gru_cell_1/add_2AddV23bidirectional/forward_gru/gru_cell_1/split:output:2,bidirectional/forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)bidirectional/forward_gru/gru_cell_1/ReluRelu.bidirectional/forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
*bidirectional/forward_gru/gru_cell_1/mul_1Mul0bidirectional/forward_gru/gru_cell_1/Sigmoid:y:0(bidirectional/forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
*bidirectional/forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
(bidirectional/forward_gru/gru_cell_1/subSub3bidirectional/forward_gru/gru_cell_1/sub/x:output:00bidirectional/forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ê
*bidirectional/forward_gru/gru_cell_1/mul_2Mul,bidirectional/forward_gru/gru_cell_1/sub:z:07bidirectional/forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Å
*bidirectional/forward_gru/gru_cell_1/add_3AddV2.bidirectional/forward_gru/gru_cell_1/mul_1:z:0.bidirectional/forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

7bidirectional/forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
)bidirectional/forward_gru/TensorArrayV2_1TensorListReserve@bidirectional/forward_gru/TensorArrayV2_1/element_shape:output:02bidirectional/forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ`
bidirectional/forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2bidirectional/forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿn
,bidirectional/forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
bidirectional/forward_gru/whileWhile5bidirectional/forward_gru/while/loop_counter:output:0;bidirectional/forward_gru/while/maximum_iterations:output:0'bidirectional/forward_gru/time:output:02bidirectional/forward_gru/TensorArrayV2_1:handle:0(bidirectional/forward_gru/zeros:output:02bidirectional/forward_gru/strided_slice_1:output:0Qbidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0<bidirectional_forward_gru_gru_cell_1_readvariableop_resourceCbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resourceEbidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *6
body.R,
*bidirectional_forward_gru_while_body_19208*6
cond.R,
*bidirectional_forward_gru_while_cond_19207*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
Jbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
<bidirectional/forward_gru/TensorArrayV2Stack/TensorListStackTensorListStack(bidirectional/forward_gru/while:output:3Sbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
/bidirectional/forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1bidirectional/forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1bidirectional/forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
)bidirectional/forward_gru/strided_slice_3StridedSliceEbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack:tensor:08bidirectional/forward_gru/strided_slice_3/stack:output:0:bidirectional/forward_gru/strided_slice_3/stack_1:output:0:bidirectional/forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
*bidirectional/forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ä
%bidirectional/forward_gru/transpose_1	TransposeEbidirectional/forward_gru/TensorArrayV2Stack/TensorListStack:tensor:03bidirectional/forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
u
!bidirectional/forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
 bidirectional/backward_gru/ShapeShapeinputs*
T0*
_output_shapes
:x
.bidirectional/backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0bidirectional/backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0bidirectional/backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
(bidirectional/backward_gru/strided_sliceStridedSlice)bidirectional/backward_gru/Shape:output:07bidirectional/backward_gru/strided_slice/stack:output:09bidirectional/backward_gru/strided_slice/stack_1:output:09bidirectional/backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)bidirectional/backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
Ä
'bidirectional/backward_gru/zeros/packedPack1bidirectional/backward_gru/strided_slice:output:02bidirectional/backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&bidirectional/backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
 bidirectional/backward_gru/zerosFill0bidirectional/backward_gru/zeros/packed:output:0/bidirectional/backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
)bidirectional/backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
$bidirectional/backward_gru/transpose	Transposeinputs2bidirectional/backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
"bidirectional/backward_gru/Shape_1Shape(bidirectional/backward_gru/transpose:y:0*
T0*
_output_shapes
:z
0bidirectional/backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*bidirectional/backward_gru/strided_slice_1StridedSlice+bidirectional/backward_gru/Shape_1:output:09bidirectional/backward_gru/strided_slice_1/stack:output:0;bidirectional/backward_gru/strided_slice_1/stack_1:output:0;bidirectional/backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6bidirectional/backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
(bidirectional/backward_gru/TensorArrayV2TensorListReserve?bidirectional/backward_gru/TensorArrayV2/element_shape:output:03bidirectional/backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)bidirectional/backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Å
$bidirectional/backward_gru/ReverseV2	ReverseV2(bidirectional/backward_gru/transpose:y:02bidirectional/backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
Pbidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¶
Bbidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor-bidirectional/backward_gru/ReverseV2:output:0Ybidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒz
0bidirectional/backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2bidirectional/backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
*bidirectional/backward_gru/strided_slice_2StridedSlice(bidirectional/backward_gru/transpose:y:09bidirectional/backward_gru/strided_slice_2/stack:output:0;bidirectional/backward_gru/strided_slice_2/stack_1:output:0;bidirectional/backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask²
4bidirectional/backward_gru/gru_cell_2/ReadVariableOpReadVariableOp=bidirectional_backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0«
-bidirectional/backward_gru/gru_cell_2/unstackUnpack<bidirectional/backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÀ
;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOpDbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0â
,bidirectional/backward_gru/gru_cell_2/MatMulMatMul3bidirectional/backward_gru/strided_slice_2:output:0Cbidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
-bidirectional/backward_gru/gru_cell_2/BiasAddBiasAdd6bidirectional/backward_gru/gru_cell_2/MatMul:product:06bidirectional/backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5bidirectional/backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+bidirectional/backward_gru/gru_cell_2/splitSplit>bidirectional/backward_gru/gru_cell_2/split/split_dim:output:06bidirectional/backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÄ
=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOpFbidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0Ü
.bidirectional/backward_gru/gru_cell_2/MatMul_1MatMul)bidirectional/backward_gru/zeros:output:0Ebidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
/bidirectional/backward_gru/gru_cell_2/BiasAdd_1BiasAdd8bidirectional/backward_gru/gru_cell_2/MatMul_1:product:06bidirectional/backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+bidirectional/backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
7bidirectional/backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÞ
-bidirectional/backward_gru/gru_cell_2/split_1SplitV8bidirectional/backward_gru/gru_cell_2/BiasAdd_1:output:04bidirectional/backward_gru/gru_cell_2/Const:output:0@bidirectional/backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÒ
)bidirectional/backward_gru/gru_cell_2/addAddV24bidirectional/backward_gru/gru_cell_2/split:output:06bidirectional/backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-bidirectional/backward_gru/gru_cell_2/SigmoidSigmoid-bidirectional/backward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô
+bidirectional/backward_gru/gru_cell_2/add_1AddV24bidirectional/backward_gru/gru_cell_2/split:output:16bidirectional/backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/bidirectional/backward_gru/gru_cell_2/Sigmoid_1Sigmoid/bidirectional/backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ï
)bidirectional/backward_gru/gru_cell_2/mulMul3bidirectional/backward_gru/gru_cell_2/Sigmoid_1:y:06bidirectional/backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ë
+bidirectional/backward_gru/gru_cell_2/add_2AddV24bidirectional/backward_gru/gru_cell_2/split:output:2-bidirectional/backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*bidirectional/backward_gru/gru_cell_2/ReluRelu/bidirectional/backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â
+bidirectional/backward_gru/gru_cell_2/mul_1Mul1bidirectional/backward_gru/gru_cell_2/Sigmoid:y:0)bidirectional/backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
+bidirectional/backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ë
)bidirectional/backward_gru/gru_cell_2/subSub4bidirectional/backward_gru/gru_cell_2/sub/x:output:01bidirectional/backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Í
+bidirectional/backward_gru/gru_cell_2/mul_2Mul-bidirectional/backward_gru/gru_cell_2/sub:z:08bidirectional/backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
È
+bidirectional/backward_gru/gru_cell_2/add_3AddV2/bidirectional/backward_gru/gru_cell_2/mul_1:z:0/bidirectional/backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

8bidirectional/backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
*bidirectional/backward_gru/TensorArrayV2_1TensorListReserveAbidirectional/backward_gru/TensorArrayV2_1/element_shape:output:03bidirectional/backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒa
bidirectional/backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3bidirectional/backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿo
-bidirectional/backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
 bidirectional/backward_gru/whileWhile6bidirectional/backward_gru/while/loop_counter:output:0<bidirectional/backward_gru/while/maximum_iterations:output:0(bidirectional/backward_gru/time:output:03bidirectional/backward_gru/TensorArrayV2_1:handle:0)bidirectional/backward_gru/zeros:output:03bidirectional/backward_gru/strided_slice_1:output:0Rbidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0=bidirectional_backward_gru_gru_cell_2_readvariableop_resourceDbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resourceFbidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *7
body/R-
+bidirectional_backward_gru_while_body_19359*7
cond/R-
+bidirectional_backward_gru_while_cond_19358*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
Kbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
=bidirectional/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStack)bidirectional/backward_gru/while:output:3Tbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
0bidirectional/backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ|
2bidirectional/backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2bidirectional/backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
*bidirectional/backward_gru/strided_slice_3StridedSliceFbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack:tensor:09bidirectional/backward_gru/strided_slice_3/stack:output:0;bidirectional/backward_gru/strided_slice_3/stack_1:output:0;bidirectional/backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
+bidirectional/backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ç
&bidirectional/backward_gru/transpose_1	TransposeFbidirectional/backward_gru/TensorArrayV2Stack/TensorListStack:tensor:04bidirectional/backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
"bidirectional/backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    [
bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :è
bidirectional/concatConcatV22bidirectional/forward_gru/strided_slice_3:output:03bidirectional/backward_gru/strided_slice_3:output:0"bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMulbidirectional/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCbidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Ó
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDbidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp<^bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp>^bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp5^bidirectional/backward_gru/gru_cell_2/ReadVariableOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^bidirectional/backward_gru/while;^bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp=^bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp4^bidirectional/forward_gru/gru_cell_1/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp ^bidirectional/forward_gru/while^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2z
;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp;bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp2~
=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp=bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2l
4bidirectional/backward_gru/gru_cell_2/ReadVariableOp4bidirectional/backward_gru/gru_cell_2/ReadVariableOp2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 bidirectional/backward_gru/while bidirectional/backward_gru/while2x
:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp:bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp2|
<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp<bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2j
3bidirectional/forward_gru/gru_cell_1/ReadVariableOp3bidirectional/forward_gru/gru_cell_1/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2B
bidirectional/forward_gru/whilebidirectional/forward_gru/while2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ


backward_gru_while_cond_204476
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_28
4backward_gru_while_less_backward_gru_strided_slice_1M
Ibackward_gru_while_backward_gru_while_cond_20447___redundant_placeholder0M
Ibackward_gru_while_backward_gru_while_cond_20447___redundant_placeholder1M
Ibackward_gru_while_backward_gru_while_cond_20447___redundant_placeholder2M
Ibackward_gru_while_backward_gru_while_cond_20447___redundant_placeholder3
backward_gru_while_identity

backward_gru/while/LessLessbackward_gru_while_placeholder4backward_gru_while_less_backward_gru_strided_slice_1*
T0*
_output_shapes
: e
backward_gru/while/IdentityIdentitybackward_gru/while/Less:z:0*
T0
*
_output_shapes
: "C
backward_gru_while_identity$backward_gru/while/Identity:output:0*(
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
§Z
×
G__inference_backward_gru_layer_call_and_return_conditional_losses_22124
inputs_04
"gru_cell_2_readvariableop_resource:;
)gru_cell_2_matmul_readvariableop_resource:=
+gru_cell_2_matmul_1_readvariableop_resource:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢while=
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
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
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
while_body_22029*
condR
while_cond_22028*8
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
 *    ¸
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÁK
à

forward_gru_while_body_209334
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_23
/forward_gru_while_forward_gru_strided_slice_1_0o
kforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0H
6forward_gru_while_gru_cell_1_readvariableop_resource_0:O
=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:Q
?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:

forward_gru_while_identity 
forward_gru_while_identity_1 
forward_gru_while_identity_2 
forward_gru_while_identity_3 
forward_gru_while_identity_41
-forward_gru_while_forward_gru_strided_slice_1m
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorF
4forward_gru_while_gru_cell_1_readvariableop_resource:M
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource:O
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢+forward_gru/while/gru_cell_1/ReadVariableOp
Cforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   â
5forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0forward_gru_while_placeholderLforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¢
+forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOp6forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
$forward_gru/while/gru_cell_1/unstackUnpack3forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num°
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
#forward_gru/while/gru_cell_1/MatMulMatMul<forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0:forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$forward_gru/while/gru_cell_1/BiasAddBiasAdd-forward_gru/while/gru_cell_1/MatMul:product:0-forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
"forward_gru/while/gru_cell_1/splitSplit5forward_gru/while/gru_cell_1/split/split_dim:output:0-forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split´
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0À
%forward_gru/while/gru_cell_1/MatMul_1MatMulforward_gru_while_placeholder_2<forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
&forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd/forward_gru/while/gru_cell_1/MatMul_1:product:0-forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿy
.forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
$forward_gru/while/gru_cell_1/split_1SplitV/forward_gru/while/gru_cell_1/BiasAdd_1:output:0+forward_gru/while/gru_cell_1/Const:output:07forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split·
 forward_gru/while/gru_cell_1/addAddV2+forward_gru/while/gru_cell_1/split:output:0-forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$forward_gru/while/gru_cell_1/SigmoidSigmoid$forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
"forward_gru/while/gru_cell_1/add_1AddV2+forward_gru/while/gru_cell_1/split:output:1-forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid&forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
 forward_gru/while/gru_cell_1/mulMul*forward_gru/while/gru_cell_1/Sigmoid_1:y:0-forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
"forward_gru/while/gru_cell_1/add_2AddV2+forward_gru/while/gru_cell_1/split:output:2$forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!forward_gru/while/gru_cell_1/ReluRelu&forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
"forward_gru/while/gru_cell_1/mul_1Mul(forward_gru/while/gru_cell_1/Sigmoid:y:0forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
"forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
 forward_gru/while/gru_cell_1/subSub+forward_gru/while/gru_cell_1/sub/x:output:0(forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
"forward_gru/while/gru_cell_1/mul_2Mul$forward_gru/while/gru_cell_1/sub:z:0/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
"forward_gru/while/gru_cell_1/add_3AddV2&forward_gru/while/gru_cell_1/mul_1:z:0&forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
6forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemforward_gru_while_placeholder_1forward_gru_while_placeholder&forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒY
forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/addAddV2forward_gru_while_placeholder forward_gru/while/add/y:output:0*
T0*
_output_shapes
: [
forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/add_1AddV20forward_gru_while_forward_gru_while_loop_counter"forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: }
forward_gru/while/IdentityIdentityforward_gru/while/add_1:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: 
forward_gru/while/Identity_1Identity6forward_gru_while_forward_gru_while_maximum_iterations^forward_gru/while/NoOp*
T0*
_output_shapes
: }
forward_gru/while/Identity_2Identityforward_gru/while/add:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: ½
forward_gru/while/Identity_3IdentityFforward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
forward_gru/while/Identity_4Identity&forward_gru/while/gru_cell_1/add_3:z:0^forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
forward_gru/while/NoOpNoOp3^forward_gru/while/gru_cell_1/MatMul/ReadVariableOp5^forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp,^forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "`
-forward_gru_while_forward_gru_strided_slice_1/forward_gru_while_forward_gru_strided_slice_1_0"
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"|
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"n
4forward_gru_while_gru_cell_1_readvariableop_resource6forward_gru_while_gru_cell_1_readvariableop_resource_0"A
forward_gru_while_identity#forward_gru/while/Identity:output:0"E
forward_gru_while_identity_1%forward_gru/while/Identity_1:output:0"E
forward_gru_while_identity_2%forward_gru/while/Identity_2:output:0"E
forward_gru_while_identity_3%forward_gru/while/Identity_3:output:0"E
forward_gru_while_identity_4%forward_gru/while/Identity_4:output:0"Ø
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2h
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2l
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2Z
+forward_gru/while/gru_cell_1/ReadVariableOp+forward_gru/while/gru_cell_1/ReadVariableOp: 
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
¹<
ø
while_body_21341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_1_readvariableop_resource_0:C
1while_gru_cell_1_matmul_readvariableop_resource_0:E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_1_readvariableop_resource:A
/while_gru_cell_1_matmul_readvariableop_resource:C
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
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
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 
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



forward_gru_while_cond_186144
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_26
2forward_gru_while_less_forward_gru_strided_slice_1K
Gforward_gru_while_forward_gru_while_cond_18614___redundant_placeholder0K
Gforward_gru_while_forward_gru_while_cond_18614___redundant_placeholder1K
Gforward_gru_while_forward_gru_while_cond_18614___redundant_placeholder2K
Gforward_gru_while_forward_gru_while_cond_18614___redundant_placeholder3
forward_gru_while_identity

forward_gru/while/LessLessforward_gru_while_placeholder2forward_gru_while_less_forward_gru_strided_slice_1*
T0*
_output_shapes
: c
forward_gru/while/IdentityIdentityforward_gru/while/Less:z:0*
T0
*
_output_shapes
: "A
forward_gru_while_identity#forward_gru/while/Identity:output:0*(
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

¼
,__inference_backward_gru_layer_call_fn_21930
inputs_0
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallë
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
GPU 2J 8 *P
fKRI
G__inference_backward_gru_layer_call_and_return_conditional_losses_17115o
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
while_body_17951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_1_readvariableop_resource_0:C
1while_gru_cell_1_matmul_readvariableop_resource_0:E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_1_readvariableop_resource:A
/while_gru_cell_1_matmul_readvariableop_resource:C
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
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
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 
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
ãL
þ

backward_gru_while_body_210846
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_25
1backward_gru_while_backward_gru_strided_slice_1_0q
mbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0I
7backward_gru_while_gru_cell_2_readvariableop_resource_0:P
>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:R
@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

backward_gru_while_identity!
backward_gru_while_identity_1!
backward_gru_while_identity_2!
backward_gru_while_identity_3!
backward_gru_while_identity_43
/backward_gru_while_backward_gru_strided_slice_1o
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensorG
5backward_gru_while_gru_cell_2_readvariableop_resource:N
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource:P
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢,backward_gru/while/gru_cell_2/ReadVariableOp
Dbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ç
6backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0backward_gru_while_placeholderMbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¤
,backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOp7backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
%backward_gru/while/gru_cell_2/unstackUnpack4backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ü
$backward_gru/while/gru_cell_2/MatMulMatMul=backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0;backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
%backward_gru/while/gru_cell_2/BiasAddBiasAdd.backward_gru/while/gru_cell_2/MatMul:product:0.backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#backward_gru/while/gru_cell_2/splitSplit6backward_gru/while/gru_cell_2/split/split_dim:output:0.backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¶
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Ã
&backward_gru/while/gru_cell_2/MatMul_1MatMul backward_gru_while_placeholder_2=backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd0backward_gru/while/gru_cell_2/MatMul_1:product:0.backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿz
/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%backward_gru/while/gru_cell_2/split_1SplitV0backward_gru/while/gru_cell_2/BiasAdd_1:output:0,backward_gru/while/gru_cell_2/Const:output:08backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
!backward_gru/while/gru_cell_2/addAddV2,backward_gru/while/gru_cell_2/split:output:0.backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%backward_gru/while/gru_cell_2/SigmoidSigmoid%backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
#backward_gru/while/gru_cell_2/add_1AddV2,backward_gru/while/gru_cell_2/split:output:1.backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid'backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
!backward_gru/while/gru_cell_2/mulMul+backward_gru/while/gru_cell_2/Sigmoid_1:y:0.backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
#backward_gru/while/gru_cell_2/add_2AddV2,backward_gru/while/gru_cell_2/split:output:2%backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"backward_gru/while/gru_cell_2/ReluRelu'backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
#backward_gru/while/gru_cell_2/mul_1Mul)backward_gru/while/gru_cell_2/Sigmoid:y:0 backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
#backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!backward_gru/while/gru_cell_2/subSub,backward_gru/while/gru_cell_2/sub/x:output:0)backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
#backward_gru/while/gru_cell_2/mul_2Mul%backward_gru/while/gru_cell_2/sub:z:00backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
#backward_gru/while/gru_cell_2/add_3AddV2'backward_gru/while/gru_cell_2/mul_1:z:0'backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
7backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem backward_gru_while_placeholder_1backward_gru_while_placeholder'backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒZ
backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/addAddV2backward_gru_while_placeholder!backward_gru/while/add/y:output:0*
T0*
_output_shapes
: \
backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/add_1AddV22backward_gru_while_backward_gru_while_loop_counter#backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru/while/IdentityIdentitybackward_gru/while/add_1:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_1Identity8backward_gru_while_backward_gru_while_maximum_iterations^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_2Identitybackward_gru/while/add:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: À
backward_gru/while/Identity_3IdentityGbackward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
backward_gru/while/Identity_4Identity'backward_gru/while/gru_cell_2/add_3:z:0^backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
backward_gru/while/NoOpNoOp4^backward_gru/while/gru_cell_2/MatMul/ReadVariableOp6^backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp-^backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/backward_gru_while_backward_gru_strided_slice_11backward_gru_while_backward_gru_strided_slice_1_0"
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"~
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"p
5backward_gru_while_gru_cell_2_readvariableop_resource7backward_gru_while_gru_cell_2_readvariableop_resource_0"C
backward_gru_while_identity$backward_gru/while/Identity:output:0"G
backward_gru_while_identity_1&backward_gru/while/Identity_1:output:0"G
backward_gru_while_identity_2&backward_gru/while/Identity_2:output:0"G
backward_gru_while_identity_3&backward_gru/while/Identity_3:output:0"G
backward_gru_while_identity_4&backward_gru/while/Identity_4:output:0"Ü
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensormbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2j
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2n
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2\
,backward_gru/while/gru_cell_2/ReadVariableOp,backward_gru/while/gru_cell_2/ReadVariableOp: 
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
°
¦
+bidirectional_backward_gru_while_cond_19358R
Nbidirectional_backward_gru_while_bidirectional_backward_gru_while_loop_counterX
Tbidirectional_backward_gru_while_bidirectional_backward_gru_while_maximum_iterations0
,bidirectional_backward_gru_while_placeholder2
.bidirectional_backward_gru_while_placeholder_12
.bidirectional_backward_gru_while_placeholder_2T
Pbidirectional_backward_gru_while_less_bidirectional_backward_gru_strided_slice_1i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19358___redundant_placeholder0i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19358___redundant_placeholder1i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19358___redundant_placeholder2i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19358___redundant_placeholder3-
)bidirectional_backward_gru_while_identity
Î
%bidirectional/backward_gru/while/LessLess,bidirectional_backward_gru_while_placeholderPbidirectional_backward_gru_while_less_bidirectional_backward_gru_strided_slice_1*
T0*
_output_shapes
: 
)bidirectional/backward_gru/while/IdentityIdentity)bidirectional/backward_gru/while/Less:z:0*
T0
*
_output_shapes
: "_
)bidirectional_backward_gru_while_identity2bidirectional/backward_gru/while/Identity:output:0*(
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
µ


backward_gru_while_cond_201296
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_28
4backward_gru_while_less_backward_gru_strided_slice_1M
Ibackward_gru_while_backward_gru_while_cond_20129___redundant_placeholder0M
Ibackward_gru_while_backward_gru_while_cond_20129___redundant_placeholder1M
Ibackward_gru_while_backward_gru_while_cond_20129___redundant_placeholder2M
Ibackward_gru_while_backward_gru_while_cond_20129___redundant_placeholder3
backward_gru_while_identity

backward_gru/while/LessLessbackward_gru_while_placeholder4backward_gru_while_less_backward_gru_strided_slice_1*
T0*
_output_shapes
: e
backward_gru/while/IdentityIdentitybackward_gru/while/Less:z:0*
T0
*
_output_shapes
: "C
backward_gru_while_identity$backward_gru/while/Identity:output:0*(
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
¹X
Õ
F__inference_forward_gru_layer_call_and_return_conditional_losses_21436
inputs_04
"gru_cell_1_readvariableop_resource:;
)gru_cell_1_matmul_readvariableop_resource:=
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while=
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
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
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
while_body_21341*
condR
while_cond_21340*8
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
 *    ·
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ïÑ
õ
H__inference_bidirectional_layer_call_and_return_conditional_losses_20551
inputs_0@
.forward_gru_gru_cell_1_readvariableop_resource:G
5forward_gru_gru_cell_1_matmul_readvariableop_resource:I
7forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
A
/backward_gru_gru_cell_2_readvariableop_resource:H
6backward_gru_gru_cell_2_matmul_readvariableop_resource:J
8backward_gru_gru_cell_2_matmul_1_readvariableop_resource:

identity¢-backward_gru/gru_cell_2/MatMul/ReadVariableOp¢/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢&backward_gru/gru_cell_2/ReadVariableOp¢backward_gru/while¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢,forward_gru/gru_cell_1/MatMul/ReadVariableOp¢.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢%forward_gru/gru_cell_1/ReadVariableOp¢forward_gru/whileI
forward_gru/ShapeShapeinputs_0*
T0*
_output_shapes
:i
forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_sliceStridedSliceforward_gru/Shape:output:0(forward_gru/strided_slice/stack:output:0*forward_gru/strided_slice/stack_1:output:0*forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru/zeros/packedPack"forward_gru/strided_slice:output:0#forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru/zerosFill!forward_gru/zeros/packed:output:0 forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru/transpose	Transposeinputs_0#forward_gru/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ\
forward_gru/Shape_1Shapeforward_gru/transpose:y:0*
T0*
_output_shapes
:k
!forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_slice_1StridedSliceforward_gru/Shape_1:output:0*forward_gru/strided_slice_1/stack:output:0,forward_gru/strided_slice_1/stack_1:output:0,forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
forward_gru/TensorArrayV2TensorListReserve0forward_gru/TensorArrayV2/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
3forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru/transpose:y:0Jforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
forward_gru/strided_slice_2StridedSliceforward_gru/transpose:y:0*forward_gru/strided_slice_2/stack:output:0,forward_gru/strided_slice_2/stack_1:output:0,forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
%forward_gru/gru_cell_1/ReadVariableOpReadVariableOp.forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0
forward_gru/gru_cell_1/unstackUnpack-forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¢
,forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
forward_gru/gru_cell_1/MatMulMatMul$forward_gru/strided_slice_2:output:04forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
forward_gru/gru_cell_1/BiasAddBiasAdd'forward_gru/gru_cell_1/MatMul:product:0'forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
forward_gru/gru_cell_1/splitSplit/forward_gru/gru_cell_1/split/split_dim:output:0'forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¦
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¯
forward_gru/gru_cell_1/MatMul_1MatMulforward_gru/zeros:output:06forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 forward_gru/gru_cell_1/BiasAdd_1BiasAdd)forward_gru/gru_cell_1/MatMul_1:product:0'forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿs
(forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
forward_gru/gru_cell_1/split_1SplitV)forward_gru/gru_cell_1/BiasAdd_1:output:0%forward_gru/gru_cell_1/Const:output:01forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¥
forward_gru/gru_cell_1/addAddV2%forward_gru/gru_cell_1/split:output:0'forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru/gru_cell_1/SigmoidSigmoidforward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
forward_gru/gru_cell_1/add_1AddV2%forward_gru/gru_cell_1/split:output:1'forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru/gru_cell_1/Sigmoid_1Sigmoid forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
forward_gru/gru_cell_1/mulMul$forward_gru/gru_cell_1/Sigmoid_1:y:0'forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_2AddV2%forward_gru/gru_cell_1/split:output:2forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
forward_gru/gru_cell_1/ReluRelu forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/mul_1Mul"forward_gru/gru_cell_1/Sigmoid:y:0forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
forward_gru/gru_cell_1/subSub%forward_gru/gru_cell_1/sub/x:output:0"forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
forward_gru/gru_cell_1/mul_2Mulforward_gru/gru_cell_1/sub:z:0)forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_3AddV2 forward_gru/gru_cell_1/mul_1:z:0 forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ü
forward_gru/TensorArrayV2_1TensorListReserve2forward_gru/TensorArrayV2_1/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
forward_gru/whileWhile'forward_gru/while/loop_counter:output:0-forward_gru/while/maximum_iterations:output:0forward_gru/time:output:0$forward_gru/TensorArrayV2_1:handle:0forward_gru/zeros:output:0$forward_gru/strided_slice_1:output:0Cforward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0.forward_gru_gru_cell_1_readvariableop_resource5forward_gru_gru_cell_1_matmul_readvariableop_resource7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *(
body R
forward_gru_while_body_20297*(
cond R
forward_gru_while_cond_20296*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
<forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ï
.forward_gru/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru/while:output:3Eforward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0t
!forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
forward_gru/strided_slice_3StridedSlice7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0*forward_gru/strided_slice_3/stack:output:0,forward_gru/strided_slice_3/stack_1:output:0,forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskq
forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
forward_gru/transpose_1	Transpose7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0%forward_gru/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
g
forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    J
backward_gru/ShapeShapeinputs_0*
T0*
_output_shapes
:j
 backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_sliceStridedSlicebackward_gru/Shape:output:0)backward_gru/strided_slice/stack:output:0+backward_gru/strided_slice/stack_1:output:0+backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

backward_gru/zeros/packedPack#backward_gru/strided_slice:output:0$backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru/zerosFill"backward_gru/zeros/packed:output:0!backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru/transpose	Transposeinputs_0$backward_gru/transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
backward_gru/Shape_1Shapebackward_gru/transpose:y:0*
T0*
_output_shapes
:l
"backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_slice_1StridedSlicebackward_gru/Shape_1:output:0+backward_gru/strided_slice_1/stack:output:0-backward_gru/strided_slice_1/stack_1:output:0-backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
backward_gru/TensorArrayV2TensorListReserve1backward_gru/TensorArrayV2/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ­
backward_gru/ReverseV2	ReverseV2backward_gru/transpose:y:0$backward_gru/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Bbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ
4backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbackward_gru/ReverseV2:output:0Kbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
backward_gru/strided_slice_2StridedSlicebackward_gru/transpose:y:0+backward_gru/strided_slice_2/stack:output:0-backward_gru/strided_slice_2/stack_1:output:0-backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
&backward_gru/gru_cell_2/ReadVariableOpReadVariableOp/backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0
backward_gru/gru_cell_2/unstackUnpack.backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¤
-backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
backward_gru/gru_cell_2/MatMulMatMul%backward_gru/strided_slice_2:output:05backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
backward_gru/gru_cell_2/BiasAddBiasAdd(backward_gru/gru_cell_2/MatMul:product:0(backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
backward_gru/gru_cell_2/splitSplit0backward_gru/gru_cell_2/split/split_dim:output:0(backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0²
 backward_gru/gru_cell_2/MatMul_1MatMulbackward_gru/zeros:output:07backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!backward_gru/gru_cell_2/BiasAdd_1BiasAdd*backward_gru/gru_cell_2/MatMul_1:product:0(backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿt
)backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
backward_gru/gru_cell_2/split_1SplitV*backward_gru/gru_cell_2/BiasAdd_1:output:0&backward_gru/gru_cell_2/Const:output:02backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
backward_gru/gru_cell_2/addAddV2&backward_gru/gru_cell_2/split:output:0(backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru/gru_cell_2/SigmoidSigmoidbackward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
backward_gru/gru_cell_2/add_1AddV2&backward_gru/gru_cell_2/split:output:1(backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru/gru_cell_2/Sigmoid_1Sigmoid!backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
backward_gru/gru_cell_2/mulMul%backward_gru/gru_cell_2/Sigmoid_1:y:0(backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
backward_gru/gru_cell_2/add_2AddV2&backward_gru/gru_cell_2/split:output:2backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
backward_gru/gru_cell_2/ReluRelu!backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/mul_1Mul#backward_gru/gru_cell_2/Sigmoid:y:0backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
backward_gru/gru_cell_2/subSub&backward_gru/gru_cell_2/sub/x:output:0#backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
backward_gru/gru_cell_2/mul_2Mulbackward_gru/gru_cell_2/sub:z:0*backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/add_3AddV2!backward_gru/gru_cell_2/mul_1:z:0!backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
*backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ß
backward_gru/TensorArrayV2_1TensorListReserve3backward_gru/TensorArrayV2_1/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
backward_gru/whileWhile(backward_gru/while/loop_counter:output:0.backward_gru/while/maximum_iterations:output:0backward_gru/time:output:0%backward_gru/TensorArrayV2_1:handle:0backward_gru/zeros:output:0%backward_gru/strided_slice_1:output:0Dbackward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/backward_gru_gru_cell_2_readvariableop_resource6backward_gru_gru_cell_2_matmul_readvariableop_resource8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *)
body!R
backward_gru_while_body_20448*)
cond!R
backward_gru_while_cond_20447*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
=backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ò
/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru/while:output:3Fbackward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
"backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
backward_gru/strided_slice_3StridedSlice8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0+backward_gru/strided_slice_3/stack:output:0-backward_gru/strided_slice_3/stack_1:output:0-backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskr
backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Æ
backward_gru/transpose_1	Transpose8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0&backward_gru/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
h
backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2$forward_gru/strided_slice_3:output:0%backward_gru/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Å
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp.^backward_gru/gru_cell_2/MatMul/ReadVariableOp0^backward_gru/gru_cell_2/MatMul_1/ReadVariableOp'^backward_gru/gru_cell_2/ReadVariableOp^backward_gru/whileO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp-^forward_gru/gru_cell_1/MatMul/ReadVariableOp/^forward_gru/gru_cell_1/MatMul_1/ReadVariableOp&^forward_gru/gru_cell_1/ReadVariableOp^forward_gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_gru/gru_cell_2/MatMul/ReadVariableOp-backward_gru/gru_cell_2/MatMul/ReadVariableOp2b
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2P
&backward_gru/gru_cell_2/ReadVariableOp&backward_gru/gru_cell_2/ReadVariableOp2(
backward_gru/whilebackward_gru/while2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2\
,forward_gru/gru_cell_1/MatMul/ReadVariableOp,forward_gru/gru_cell_1/MatMul/ReadVariableOp2`
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2N
%forward_gru/gru_cell_1/ReadVariableOp%forward_gru/gru_cell_1/ReadVariableOp2&
forward_gru/whileforward_gru/while:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
?
Î
F__inference_forward_gru_layer_call_and_return_conditional_losses_16945

inputs"
gru_cell_1_16863:"
gru_cell_1_16865:"
gru_cell_1_16867:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_1/StatefulPartitionedCall¢while;
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
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_16863gru_cell_1_16865gru_cell_1_16867*
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16823n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_16863gru_cell_1_16865gru_cell_1_16867*
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
while_body_16875*
condR
while_cond_16874*8
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
 *    
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_1_16865*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*bidirectional_forward_gru_while_cond_19539P
Lbidirectional_forward_gru_while_bidirectional_forward_gru_while_loop_counterV
Rbidirectional_forward_gru_while_bidirectional_forward_gru_while_maximum_iterations/
+bidirectional_forward_gru_while_placeholder1
-bidirectional_forward_gru_while_placeholder_11
-bidirectional_forward_gru_while_placeholder_2R
Nbidirectional_forward_gru_while_less_bidirectional_forward_gru_strided_slice_1g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19539___redundant_placeholder0g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19539___redundant_placeholder1g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19539___redundant_placeholder2g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19539___redundant_placeholder3,
(bidirectional_forward_gru_while_identity
Ê
$bidirectional/forward_gru/while/LessLess+bidirectional_forward_gru_while_placeholderNbidirectional_forward_gru_while_less_bidirectional_forward_gru_strided_slice_1*
T0*
_output_shapes
: 
(bidirectional/forward_gru/while/IdentityIdentity(bidirectional/forward_gru/while/Less:z:0*
T0
*
_output_shapes
: "]
(bidirectional_forward_gru_while_identity1bidirectional/forward_gru/while/Identity:output:0*(
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
while_body_17770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:C
1while_gru_cell_2_matmul_readvariableop_resource_0:E
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:A
/while_gru_cell_2_matmul_readvariableop_resource:C
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
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
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp: 
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
A
Ð
G__inference_backward_gru_layer_call_and_return_conditional_losses_17115

inputs"
gru_cell_2_17033:"
gru_cell_2_17035:"
gru_cell_2_17037:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_2/StatefulPartitionedCall¢while;
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
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_17033gru_cell_2_17035gru_cell_2_17037*
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17032n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_17033gru_cell_2_17035gru_cell_2_17037*
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
while_body_17045*
condR
while_cond_17044*8
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
 *    
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_2_17035*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ä
NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs



forward_gru_while_cond_209324
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_26
2forward_gru_while_less_forward_gru_strided_slice_1K
Gforward_gru_while_forward_gru_while_cond_20932___redundant_placeholder0K
Gforward_gru_while_forward_gru_while_cond_20932___redundant_placeholder1K
Gforward_gru_while_forward_gru_while_cond_20932___redundant_placeholder2K
Gforward_gru_while_forward_gru_while_cond_20932___redundant_placeholder3
forward_gru_while_identity

forward_gru/while/LessLessforward_gru_while_placeholder2forward_gru_while_less_forward_gru_strided_slice_1*
T0*
_output_shapes
: c
forward_gru/while/IdentityIdentityforward_gru/while/Less:z:0*
T0
*
_output_shapes
: "A
forward_gru_while_identity#forward_gru/while/Identity:output:0*(
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
¡
É
 __inference__wrapped_model_16592
bidirectional_inputY
Gsequential_bidirectional_forward_gru_gru_cell_1_readvariableop_resource:`
Nsequential_bidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource:b
Psequential_bidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
Z
Hsequential_bidirectional_backward_gru_gru_cell_2_readvariableop_resource:a
Osequential_bidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource:c
Qsequential_bidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource:
A
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:
identity¢Fsequential/bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp¢Hsequential/bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢?sequential/bidirectional/backward_gru/gru_cell_2/ReadVariableOp¢+sequential/bidirectional/backward_gru/while¢Esequential/bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp¢Gsequential/bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢>sequential/bidirectional/forward_gru/gru_cell_1/ReadVariableOp¢*sequential/bidirectional/forward_gru/while¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOpm
*sequential/bidirectional/forward_gru/ShapeShapebidirectional_input*
T0*
_output_shapes
:
8sequential/bidirectional/forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:sequential/bidirectional/forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:sequential/bidirectional/forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2sequential/bidirectional/forward_gru/strided_sliceStridedSlice3sequential/bidirectional/forward_gru/Shape:output:0Asequential/bidirectional/forward_gru/strided_slice/stack:output:0Csequential/bidirectional/forward_gru/strided_slice/stack_1:output:0Csequential/bidirectional/forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3sequential/bidirectional/forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
â
1sequential/bidirectional/forward_gru/zeros/packedPack;sequential/bidirectional/forward_gru/strided_slice:output:0<sequential/bidirectional/forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/bidirectional/forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Û
*sequential/bidirectional/forward_gru/zerosFill:sequential/bidirectional/forward_gru/zeros/packed:output:09sequential/bidirectional/forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

3sequential/bidirectional/forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ä
.sequential/bidirectional/forward_gru/transpose	Transposebidirectional_input<sequential/bidirectional/forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential/bidirectional/forward_gru/Shape_1Shape2sequential/bidirectional/forward_gru/transpose:y:0*
T0*
_output_shapes
:
:sequential/bidirectional/forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential/bidirectional/forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential/bidirectional/forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential/bidirectional/forward_gru/strided_slice_1StridedSlice5sequential/bidirectional/forward_gru/Shape_1:output:0Csequential/bidirectional/forward_gru/strided_slice_1/stack:output:0Esequential/bidirectional/forward_gru/strided_slice_1/stack_1:output:0Esequential/bidirectional/forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential/bidirectional/forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
2sequential/bidirectional/forward_gru/TensorArrayV2TensorListReserveIsequential/bidirectional/forward_gru/TensorArrayV2/element_shape:output:0=sequential/bidirectional/forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ«
Zsequential/bidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
Lsequential/bidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor2sequential/bidirectional/forward_gru/transpose:y:0csequential/bidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
:sequential/bidirectional/forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential/bidirectional/forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential/bidirectional/forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
4sequential/bidirectional/forward_gru/strided_slice_2StridedSlice2sequential/bidirectional/forward_gru/transpose:y:0Csequential/bidirectional/forward_gru/strided_slice_2/stack:output:0Esequential/bidirectional/forward_gru/strided_slice_2/stack_1:output:0Esequential/bidirectional/forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÆ
>sequential/bidirectional/forward_gru/gru_cell_1/ReadVariableOpReadVariableOpGsequential_bidirectional_forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0¿
7sequential/bidirectional/forward_gru/gru_cell_1/unstackUnpackFsequential/bidirectional/forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÔ
Esequential/bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOpNsequential_bidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
6sequential/bidirectional/forward_gru/gru_cell_1/MatMulMatMul=sequential/bidirectional/forward_gru/strided_slice_2:output:0Msequential/bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
7sequential/bidirectional/forward_gru/gru_cell_1/BiasAddBiasAdd@sequential/bidirectional/forward_gru/gru_cell_1/MatMul:product:0@sequential/bidirectional/forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential/bidirectional/forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ³
5sequential/bidirectional/forward_gru/gru_cell_1/splitSplitHsequential/bidirectional/forward_gru/gru_cell_1/split/split_dim:output:0@sequential/bidirectional/forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitØ
Gsequential/bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpPsequential_bidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0ú
8sequential/bidirectional/forward_gru/gru_cell_1/MatMul_1MatMul3sequential/bidirectional/forward_gru/zeros:output:0Osequential/bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
9sequential/bidirectional/forward_gru/gru_cell_1/BiasAdd_1BiasAddBsequential/bidirectional/forward_gru/gru_cell_1/MatMul_1:product:0@sequential/bidirectional/forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5sequential/bidirectional/forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Asequential/bidirectional/forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
7sequential/bidirectional/forward_gru/gru_cell_1/split_1SplitVBsequential/bidirectional/forward_gru/gru_cell_1/BiasAdd_1:output:0>sequential/bidirectional/forward_gru/gru_cell_1/Const:output:0Jsequential/bidirectional/forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitð
3sequential/bidirectional/forward_gru/gru_cell_1/addAddV2>sequential/bidirectional/forward_gru/gru_cell_1/split:output:0@sequential/bidirectional/forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
7sequential/bidirectional/forward_gru/gru_cell_1/SigmoidSigmoid7sequential/bidirectional/forward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
5sequential/bidirectional/forward_gru/gru_cell_1/add_1AddV2>sequential/bidirectional/forward_gru/gru_cell_1/split:output:1@sequential/bidirectional/forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
9sequential/bidirectional/forward_gru/gru_cell_1/Sigmoid_1Sigmoid9sequential/bidirectional/forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
í
3sequential/bidirectional/forward_gru/gru_cell_1/mulMul=sequential/bidirectional/forward_gru/gru_cell_1/Sigmoid_1:y:0@sequential/bidirectional/forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
é
5sequential/bidirectional/forward_gru/gru_cell_1/add_2AddV2>sequential/bidirectional/forward_gru/gru_cell_1/split:output:27sequential/bidirectional/forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
4sequential/bidirectional/forward_gru/gru_cell_1/ReluRelu9sequential/bidirectional/forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
à
5sequential/bidirectional/forward_gru/gru_cell_1/mul_1Mul;sequential/bidirectional/forward_gru/gru_cell_1/Sigmoid:y:03sequential/bidirectional/forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
5sequential/bidirectional/forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?é
3sequential/bidirectional/forward_gru/gru_cell_1/subSub>sequential/bidirectional/forward_gru/gru_cell_1/sub/x:output:0;sequential/bidirectional/forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë
5sequential/bidirectional/forward_gru/gru_cell_1/mul_2Mul7sequential/bidirectional/forward_gru/gru_cell_1/sub:z:0Bsequential/bidirectional/forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
5sequential/bidirectional/forward_gru/gru_cell_1/add_3AddV29sequential/bidirectional/forward_gru/gru_cell_1/mul_1:z:09sequential/bidirectional/forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Bsequential/bidirectional/forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   §
4sequential/bidirectional/forward_gru/TensorArrayV2_1TensorListReserveKsequential/bidirectional/forward_gru/TensorArrayV2_1/element_shape:output:0=sequential/bidirectional/forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
)sequential/bidirectional/forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 
=sequential/bidirectional/forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿy
7sequential/bidirectional/forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
*sequential/bidirectional/forward_gru/whileWhile@sequential/bidirectional/forward_gru/while/loop_counter:output:0Fsequential/bidirectional/forward_gru/while/maximum_iterations:output:02sequential/bidirectional/forward_gru/time:output:0=sequential/bidirectional/forward_gru/TensorArrayV2_1:handle:03sequential/bidirectional/forward_gru/zeros:output:0=sequential/bidirectional/forward_gru/strided_slice_1:output:0\sequential/bidirectional/forward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0Gsequential_bidirectional_forward_gru_gru_cell_1_readvariableop_resourceNsequential_bidirectional_forward_gru_gru_cell_1_matmul_readvariableop_resourcePsequential_bidirectional_forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *A
body9R7
5sequential_bidirectional_forward_gru_while_body_16336*A
cond9R7
5sequential_bidirectional_forward_gru_while_cond_16335*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations ¦
Usequential/bidirectional/forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ±
Gsequential/bidirectional/forward_gru/TensorArrayV2Stack/TensorListStackTensorListStack3sequential/bidirectional/forward_gru/while:output:3^sequential/bidirectional/forward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
:sequential/bidirectional/forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
<sequential/bidirectional/forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<sequential/bidirectional/forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
4sequential/bidirectional/forward_gru/strided_slice_3StridedSlicePsequential/bidirectional/forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0Csequential/bidirectional/forward_gru/strided_slice_3/stack:output:0Esequential/bidirectional/forward_gru/strided_slice_3/stack_1:output:0Esequential/bidirectional/forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
5sequential/bidirectional/forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
0sequential/bidirectional/forward_gru/transpose_1	TransposePsequential/bidirectional/forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0>sequential/bidirectional/forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

,sequential/bidirectional/forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
+sequential/bidirectional/backward_gru/ShapeShapebidirectional_input*
T0*
_output_shapes
:
9sequential/bidirectional/backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;sequential/bidirectional/backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;sequential/bidirectional/backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3sequential/bidirectional/backward_gru/strided_sliceStridedSlice4sequential/bidirectional/backward_gru/Shape:output:0Bsequential/bidirectional/backward_gru/strided_slice/stack:output:0Dsequential/bidirectional/backward_gru/strided_slice/stack_1:output:0Dsequential/bidirectional/backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4sequential/bidirectional/backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
å
2sequential/bidirectional/backward_gru/zeros/packedPack<sequential/bidirectional/backward_gru/strided_slice:output:0=sequential/bidirectional/backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1sequential/bidirectional/backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Þ
+sequential/bidirectional/backward_gru/zerosFill;sequential/bidirectional/backward_gru/zeros/packed:output:0:sequential/bidirectional/backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

4sequential/bidirectional/backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Æ
/sequential/bidirectional/backward_gru/transpose	Transposebidirectional_input=sequential/bidirectional/backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-sequential/bidirectional/backward_gru/Shape_1Shape3sequential/bidirectional/backward_gru/transpose:y:0*
T0*
_output_shapes
:
;sequential/bidirectional/backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential/bidirectional/backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential/bidirectional/backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/bidirectional/backward_gru/strided_slice_1StridedSlice6sequential/bidirectional/backward_gru/Shape_1:output:0Dsequential/bidirectional/backward_gru/strided_slice_1/stack:output:0Fsequential/bidirectional/backward_gru/strided_slice_1/stack_1:output:0Fsequential/bidirectional/backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Asequential/bidirectional/backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
3sequential/bidirectional/backward_gru/TensorArrayV2TensorListReserveJsequential/bidirectional/backward_gru/TensorArrayV2/element_shape:output:0>sequential/bidirectional/backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ~
4sequential/bidirectional/backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: æ
/sequential/bidirectional/backward_gru/ReverseV2	ReverseV23sequential/bidirectional/backward_gru/transpose:y:0=sequential/bidirectional/backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
[sequential/bidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
Msequential/bidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor8sequential/bidirectional/backward_gru/ReverseV2:output:0dsequential/bidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;sequential/bidirectional/backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential/bidirectional/backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential/bidirectional/backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:§
5sequential/bidirectional/backward_gru/strided_slice_2StridedSlice3sequential/bidirectional/backward_gru/transpose:y:0Dsequential/bidirectional/backward_gru/strided_slice_2/stack:output:0Fsequential/bidirectional/backward_gru/strided_slice_2/stack_1:output:0Fsequential/bidirectional/backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÈ
?sequential/bidirectional/backward_gru/gru_cell_2/ReadVariableOpReadVariableOpHsequential_bidirectional_backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0Á
8sequential/bidirectional/backward_gru/gru_cell_2/unstackUnpackGsequential/bidirectional/backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÖ
Fsequential/bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOpOsequential_bidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
7sequential/bidirectional/backward_gru/gru_cell_2/MatMulMatMul>sequential/bidirectional/backward_gru/strided_slice_2:output:0Nsequential/bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
8sequential/bidirectional/backward_gru/gru_cell_2/BiasAddBiasAddAsequential/bidirectional/backward_gru/gru_cell_2/MatMul:product:0Asequential/bidirectional/backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/bidirectional/backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¶
6sequential/bidirectional/backward_gru/gru_cell_2/splitSplitIsequential/bidirectional/backward_gru/gru_cell_2/split/split_dim:output:0Asequential/bidirectional/backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÚ
Hsequential/bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOpQsequential_bidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0ý
9sequential/bidirectional/backward_gru/gru_cell_2/MatMul_1MatMul4sequential/bidirectional/backward_gru/zeros:output:0Psequential/bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
:sequential/bidirectional/backward_gru/gru_cell_2/BiasAdd_1BiasAddCsequential/bidirectional/backward_gru/gru_cell_2/MatMul_1:product:0Asequential/bidirectional/backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6sequential/bidirectional/backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Bsequential/bidirectional/backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
8sequential/bidirectional/backward_gru/gru_cell_2/split_1SplitVCsequential/bidirectional/backward_gru/gru_cell_2/BiasAdd_1:output:0?sequential/bidirectional/backward_gru/gru_cell_2/Const:output:0Ksequential/bidirectional/backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitó
4sequential/bidirectional/backward_gru/gru_cell_2/addAddV2?sequential/bidirectional/backward_gru/gru_cell_2/split:output:0Asequential/bidirectional/backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
8sequential/bidirectional/backward_gru/gru_cell_2/SigmoidSigmoid8sequential/bidirectional/backward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
õ
6sequential/bidirectional/backward_gru/gru_cell_2/add_1AddV2?sequential/bidirectional/backward_gru/gru_cell_2/split:output:1Asequential/bidirectional/backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
:sequential/bidirectional/backward_gru/gru_cell_2/Sigmoid_1Sigmoid:sequential/bidirectional/backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ð
4sequential/bidirectional/backward_gru/gru_cell_2/mulMul>sequential/bidirectional/backward_gru/gru_cell_2/Sigmoid_1:y:0Asequential/bidirectional/backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ì
6sequential/bidirectional/backward_gru/gru_cell_2/add_2AddV2?sequential/bidirectional/backward_gru/gru_cell_2/split:output:28sequential/bidirectional/backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
5sequential/bidirectional/backward_gru/gru_cell_2/ReluRelu:sequential/bidirectional/backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ã
6sequential/bidirectional/backward_gru/gru_cell_2/mul_1Mul<sequential/bidirectional/backward_gru/gru_cell_2/Sigmoid:y:04sequential/bidirectional/backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
6sequential/bidirectional/backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ì
4sequential/bidirectional/backward_gru/gru_cell_2/subSub?sequential/bidirectional/backward_gru/gru_cell_2/sub/x:output:0<sequential/bidirectional/backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
6sequential/bidirectional/backward_gru/gru_cell_2/mul_2Mul8sequential/bidirectional/backward_gru/gru_cell_2/sub:z:0Csequential/bidirectional/backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
é
6sequential/bidirectional/backward_gru/gru_cell_2/add_3AddV2:sequential/bidirectional/backward_gru/gru_cell_2/mul_1:z:0:sequential/bidirectional/backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Csequential/bidirectional/backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ª
5sequential/bidirectional/backward_gru/TensorArrayV2_1TensorListReserveLsequential/bidirectional/backward_gru/TensorArrayV2_1/element_shape:output:0>sequential/bidirectional/backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
*sequential/bidirectional/backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 
>sequential/bidirectional/backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿz
8sequential/bidirectional/backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : §	
+sequential/bidirectional/backward_gru/whileWhileAsequential/bidirectional/backward_gru/while/loop_counter:output:0Gsequential/bidirectional/backward_gru/while/maximum_iterations:output:03sequential/bidirectional/backward_gru/time:output:0>sequential/bidirectional/backward_gru/TensorArrayV2_1:handle:04sequential/bidirectional/backward_gru/zeros:output:0>sequential/bidirectional/backward_gru/strided_slice_1:output:0]sequential/bidirectional/backward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0Hsequential_bidirectional_backward_gru_gru_cell_2_readvariableop_resourceOsequential_bidirectional_backward_gru_gru_cell_2_matmul_readvariableop_resourceQsequential_bidirectional_backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *B
body:R8
6sequential_bidirectional_backward_gru_while_body_16487*B
cond:R8
6sequential_bidirectional_backward_gru_while_cond_16486*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations §
Vsequential/bidirectional/backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ´
Hsequential/bidirectional/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStack4sequential/bidirectional/backward_gru/while:output:3_sequential/bidirectional/backward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
;sequential/bidirectional/backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=sequential/bidirectional/backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=sequential/bidirectional/backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
5sequential/bidirectional/backward_gru/strided_slice_3StridedSliceQsequential/bidirectional/backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0Dsequential/bidirectional/backward_gru/strided_slice_3/stack:output:0Fsequential/bidirectional/backward_gru/strided_slice_3/stack_1:output:0Fsequential/bidirectional/backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask
6sequential/bidirectional/backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
1sequential/bidirectional/backward_gru/transpose_1	TransposeQsequential/bidirectional/backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0?sequential/bidirectional/backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

-sequential/bidirectional/backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    f
$sequential/bidirectional/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
sequential/bidirectional/concatConcatV2=sequential/bidirectional/forward_gru/strided_slice_3:output:0>sequential/bidirectional/backward_gru/strided_slice_3:output:0-sequential/bidirectional/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0­
sequential/dense/MatMulMatMul(sequential/bidirectional/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¬
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
IdentityIdentity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
NoOpNoOpG^sequential/bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOpI^sequential/bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp@^sequential/bidirectional/backward_gru/gru_cell_2/ReadVariableOp,^sequential/bidirectional/backward_gru/whileF^sequential/bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOpH^sequential/bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp?^sequential/bidirectional/forward_gru/gru_cell_1/ReadVariableOp+^sequential/bidirectional/forward_gru/while(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2
Fsequential/bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOpFsequential/bidirectional/backward_gru/gru_cell_2/MatMul/ReadVariableOp2
Hsequential/bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpHsequential/bidirectional/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2
?sequential/bidirectional/backward_gru/gru_cell_2/ReadVariableOp?sequential/bidirectional/backward_gru/gru_cell_2/ReadVariableOp2Z
+sequential/bidirectional/backward_gru/while+sequential/bidirectional/backward_gru/while2
Esequential/bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOpEsequential/bidirectional/forward_gru/gru_cell_1/MatMul/ReadVariableOp2
Gsequential/bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOpGsequential/bidirectional/forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2
>sequential/bidirectional/forward_gru/gru_cell_1/ReadVariableOp>sequential/bidirectional/forward_gru/gru_cell_1/ReadVariableOp2X
*sequential/bidirectional/forward_gru/while*sequential/bidirectional/forward_gru/while2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namebidirectional_input
µ


backward_gru_while_cond_183246
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_28
4backward_gru_while_less_backward_gru_strided_slice_1M
Ibackward_gru_while_backward_gru_while_cond_18324___redundant_placeholder0M
Ibackward_gru_while_backward_gru_while_cond_18324___redundant_placeholder1M
Ibackward_gru_while_backward_gru_while_cond_18324___redundant_placeholder2M
Ibackward_gru_while_backward_gru_while_cond_18324___redundant_placeholder3
backward_gru_while_identity

backward_gru/while/LessLessbackward_gru_while_placeholder4backward_gru_while_less_backward_gru_strided_slice_1*
T0*
_output_shapes
: e
backward_gru/while/IdentityIdentitybackward_gru/while/Less:z:0*
T0
*
_output_shapes
: "C
backward_gru_while_identity$backward_gru/while/Identity:output:0*(
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
²^
¦
+bidirectional_backward_gru_while_body_19691R
Nbidirectional_backward_gru_while_bidirectional_backward_gru_while_loop_counterX
Tbidirectional_backward_gru_while_bidirectional_backward_gru_while_maximum_iterations0
,bidirectional_backward_gru_while_placeholder2
.bidirectional_backward_gru_while_placeholder_12
.bidirectional_backward_gru_while_placeholder_2Q
Mbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1_0
bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0W
Ebidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0:^
Lbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:`
Nbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:
-
)bidirectional_backward_gru_while_identity/
+bidirectional_backward_gru_while_identity_1/
+bidirectional_backward_gru_while_identity_2/
+bidirectional_backward_gru_while_identity_3/
+bidirectional_backward_gru_while_identity_4O
Kbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1
bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensorU
Cbidirectional_backward_gru_while_gru_cell_2_readvariableop_resource:\
Jbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource:^
Lbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢Abidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢Cbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp£
Rbidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ®
Dbidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItembidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0,bidirectional_backward_gru_while_placeholder[bidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0À
:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOpEbidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0·
3bidirectional/backward_gru/while/gru_cell_2/unstackUnpackBbidirectional/backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÎ
Abidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOpLbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0
2bidirectional/backward_gru/while/gru_cell_2/MatMulMatMulKbidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0Ibidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
3bidirectional/backward_gru/while/gru_cell_2/BiasAddBiasAdd<bidirectional/backward_gru/while/gru_cell_2/MatMul:product:0<bidirectional/backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;bidirectional/backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
1bidirectional/backward_gru/while/gru_cell_2/splitSplitDbidirectional/backward_gru/while/gru_cell_2/split/split_dim:output:0<bidirectional/backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitÒ
Cbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOpNbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0í
4bidirectional/backward_gru/while/gru_cell_2/MatMul_1MatMul.bidirectional_backward_gru_while_placeholder_2Kbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
5bidirectional/backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd>bidirectional/backward_gru/while/gru_cell_2/MatMul_1:product:0<bidirectional/backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1bidirectional/backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
=bidirectional/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
3bidirectional/backward_gru/while/gru_cell_2/split_1SplitV>bidirectional/backward_gru/while/gru_cell_2/BiasAdd_1:output:0:bidirectional/backward_gru/while/gru_cell_2/Const:output:0Fbidirectional/backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitä
/bidirectional/backward_gru/while/gru_cell_2/addAddV2:bidirectional/backward_gru/while/gru_cell_2/split:output:0<bidirectional/backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
3bidirectional/backward_gru/while/gru_cell_2/SigmoidSigmoid3bidirectional/backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
æ
1bidirectional/backward_gru/while/gru_cell_2/add_1AddV2:bidirectional/backward_gru/while/gru_cell_2/split:output:1<bidirectional/backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
5bidirectional/backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid5bidirectional/backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
á
/bidirectional/backward_gru/while/gru_cell_2/mulMul9bidirectional/backward_gru/while/gru_cell_2/Sigmoid_1:y:0<bidirectional/backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
1bidirectional/backward_gru/while/gru_cell_2/add_2AddV2:bidirectional/backward_gru/while/gru_cell_2/split:output:23bidirectional/backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
0bidirectional/backward_gru/while/gru_cell_2/ReluRelu5bidirectional/backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ó
1bidirectional/backward_gru/while/gru_cell_2/mul_1Mul7bidirectional/backward_gru/while/gru_cell_2/Sigmoid:y:0.bidirectional_backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
1bidirectional/backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ý
/bidirectional/backward_gru/while/gru_cell_2/subSub:bidirectional/backward_gru/while/gru_cell_2/sub/x:output:07bidirectional/backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
1bidirectional/backward_gru/while/gru_cell_2/mul_2Mul3bidirectional/backward_gru/while/gru_cell_2/sub:z:0>bidirectional/backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
1bidirectional/backward_gru/while/gru_cell_2/add_3AddV25bidirectional/backward_gru/while/gru_cell_2/mul_1:z:05bidirectional/backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¯
Ebidirectional/backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.bidirectional_backward_gru_while_placeholder_1,bidirectional_backward_gru_while_placeholder5bidirectional/backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒh
&bidirectional/backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :­
$bidirectional/backward_gru/while/addAddV2,bidirectional_backward_gru_while_placeholder/bidirectional/backward_gru/while/add/y:output:0*
T0*
_output_shapes
: j
(bidirectional/backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ó
&bidirectional/backward_gru/while/add_1AddV2Nbidirectional_backward_gru_while_bidirectional_backward_gru_while_loop_counter1bidirectional/backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: ª
)bidirectional/backward_gru/while/IdentityIdentity*bidirectional/backward_gru/while/add_1:z:0&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: Ö
+bidirectional/backward_gru/while/Identity_1IdentityTbidirectional_backward_gru_while_bidirectional_backward_gru_while_maximum_iterations&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: ª
+bidirectional/backward_gru/while/Identity_2Identity(bidirectional/backward_gru/while/add:z:0&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: ê
+bidirectional/backward_gru/while/Identity_3IdentityUbidirectional/backward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒÈ
+bidirectional/backward_gru/while/Identity_4Identity5bidirectional/backward_gru/while/gru_cell_2/add_3:z:0&^bidirectional/backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
®
%bidirectional/backward_gru/while/NoOpNoOpB^bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpD^bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp;^bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Kbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1Mbidirectional_backward_gru_while_bidirectional_backward_gru_strided_slice_1_0"
Lbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resourceNbidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"
Jbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resourceLbidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"
Cbidirectional_backward_gru_while_gru_cell_2_readvariableop_resourceEbidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0"_
)bidirectional_backward_gru_while_identity2bidirectional/backward_gru/while/Identity:output:0"c
+bidirectional_backward_gru_while_identity_14bidirectional/backward_gru/while/Identity_1:output:0"c
+bidirectional_backward_gru_while_identity_24bidirectional/backward_gru/while/Identity_2:output:0"c
+bidirectional_backward_gru_while_identity_34bidirectional/backward_gru/while/Identity_3:output:0"c
+bidirectional_backward_gru_while_identity_44bidirectional/backward_gru/while/Identity_4:output:0"
bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensorbidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Abidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpAbidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2
Cbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpCbidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2x
:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp:bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp: 
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
ÍZ
Õ
G__inference_backward_gru_layer_call_and_return_conditional_losses_17653

inputs4
"gru_cell_2_readvariableop_resource:;
)gru_cell_2_matmul_readvariableop_resource:=
+gru_cell_2_matmul_1_readvariableop_resource:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢while;
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
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
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
while_body_17558*
condR
while_cond_17557*8
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
 *    ¸
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_22511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22511___redundant_placeholder03
/while_while_cond_22511___redundant_placeholder13
/while_while_cond_22511___redundant_placeholder23
/while_while_cond_22511___redundant_placeholder3
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
ôj
Ò
5sequential_bidirectional_forward_gru_while_body_16336f
bsequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_loop_counterl
hsequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_maximum_iterations:
6sequential_bidirectional_forward_gru_while_placeholder<
8sequential_bidirectional_forward_gru_while_placeholder_1<
8sequential_bidirectional_forward_gru_while_placeholder_2e
asequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_strided_slice_1_0¢
sequential_bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0a
Osequential_bidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0:h
Vsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:j
Xsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:
7
3sequential_bidirectional_forward_gru_while_identity9
5sequential_bidirectional_forward_gru_while_identity_19
5sequential_bidirectional_forward_gru_while_identity_29
5sequential_bidirectional_forward_gru_while_identity_39
5sequential_bidirectional_forward_gru_while_identity_4c
_sequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_strided_slice_1 
sequential_bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_
Msequential_bidirectional_forward_gru_while_gru_cell_1_readvariableop_resource:f
Tsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource:h
Vsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢Ksequential/bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢Msequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢Dsequential/bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp­
\sequential/bidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
Nsequential/bidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_06sequential_bidirectional_forward_gru_while_placeholderesequential/bidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ô
Dsequential/bidirectional/forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOpOsequential_bidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Ë
=sequential/bidirectional/forward_gru/while/gru_cell_1/unstackUnpackLsequential/bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numâ
Ksequential/bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOpVsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0¤
<sequential/bidirectional/forward_gru/while/gru_cell_1/MatMulMatMulUsequential/bidirectional/forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0Ssequential/bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=sequential/bidirectional/forward_gru/while/gru_cell_1/BiasAddBiasAddFsequential/bidirectional/forward_gru/while/gru_cell_1/MatMul:product:0Fsequential/bidirectional/forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential/bidirectional/forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
;sequential/bidirectional/forward_gru/while/gru_cell_1/splitSplitNsequential/bidirectional/forward_gru/while/gru_cell_1/split/split_dim:output:0Fsequential/bidirectional/forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitæ
Msequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpXsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
>sequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1MatMul8sequential_bidirectional_forward_gru_while_placeholder_2Usequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?sequential/bidirectional/forward_gru/while/gru_cell_1/BiasAdd_1BiasAddHsequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1:product:0Fsequential/bidirectional/forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;sequential/bidirectional/forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Gsequential/bidirectional/forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
=sequential/bidirectional/forward_gru/while/gru_cell_1/split_1SplitVHsequential/bidirectional/forward_gru/while/gru_cell_1/BiasAdd_1:output:0Dsequential/bidirectional/forward_gru/while/gru_cell_1/Const:output:0Psequential/bidirectional/forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
9sequential/bidirectional/forward_gru/while/gru_cell_1/addAddV2Dsequential/bidirectional/forward_gru/while/gru_cell_1/split:output:0Fsequential/bidirectional/forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
=sequential/bidirectional/forward_gru/while/gru_cell_1/SigmoidSigmoid=sequential/bidirectional/forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;sequential/bidirectional/forward_gru/while/gru_cell_1/add_1AddV2Dsequential/bidirectional/forward_gru/while/gru_cell_1/split:output:1Fsequential/bidirectional/forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
?sequential/bidirectional/forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid?sequential/bidirectional/forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÿ
9sequential/bidirectional/forward_gru/while/gru_cell_1/mulMulCsequential/bidirectional/forward_gru/while/gru_cell_1/Sigmoid_1:y:0Fsequential/bidirectional/forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
;sequential/bidirectional/forward_gru/while/gru_cell_1/add_2AddV2Dsequential/bidirectional/forward_gru/while/gru_cell_1/split:output:2=sequential/bidirectional/forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
:sequential/bidirectional/forward_gru/while/gru_cell_1/ReluRelu?sequential/bidirectional/forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ñ
;sequential/bidirectional/forward_gru/while/gru_cell_1/mul_1MulAsequential/bidirectional/forward_gru/while/gru_cell_1/Sigmoid:y:08sequential_bidirectional_forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;sequential/bidirectional/forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?û
9sequential/bidirectional/forward_gru/while/gru_cell_1/subSubDsequential/bidirectional/forward_gru/while/gru_cell_1/sub/x:output:0Asequential/bidirectional/forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ý
;sequential/bidirectional/forward_gru/while/gru_cell_1/mul_2Mul=sequential/bidirectional/forward_gru/while/gru_cell_1/sub:z:0Hsequential/bidirectional/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ø
;sequential/bidirectional/forward_gru/while/gru_cell_1/add_3AddV2?sequential/bidirectional/forward_gru/while/gru_cell_1/mul_1:z:0?sequential/bidirectional/forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
Osequential/bidirectional/forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem8sequential_bidirectional_forward_gru_while_placeholder_16sequential_bidirectional_forward_gru_while_placeholder?sequential/bidirectional/forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒr
0sequential/bidirectional/forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ë
.sequential/bidirectional/forward_gru/while/addAddV26sequential_bidirectional_forward_gru_while_placeholder9sequential/bidirectional/forward_gru/while/add/y:output:0*
T0*
_output_shapes
: t
2sequential/bidirectional/forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :û
0sequential/bidirectional/forward_gru/while/add_1AddV2bsequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_loop_counter;sequential/bidirectional/forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: È
3sequential/bidirectional/forward_gru/while/IdentityIdentity4sequential/bidirectional/forward_gru/while/add_1:z:00^sequential/bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: þ
5sequential/bidirectional/forward_gru/while/Identity_1Identityhsequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_maximum_iterations0^sequential/bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: È
5sequential/bidirectional/forward_gru/while/Identity_2Identity2sequential/bidirectional/forward_gru/while/add:z:00^sequential/bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: 
5sequential/bidirectional/forward_gru/while/Identity_3Identity_sequential/bidirectional/forward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:00^sequential/bidirectional/forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒæ
5sequential/bidirectional/forward_gru/while/Identity_4Identity?sequential/bidirectional/forward_gru/while/gru_cell_1/add_3:z:00^sequential/bidirectional/forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
/sequential/bidirectional/forward_gru/while/NoOpNoOpL^sequential/bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpN^sequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpE^sequential/bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "²
Vsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resourceXsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"®
Tsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resourceVsequential_bidirectional_forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0" 
Msequential_bidirectional_forward_gru_while_gru_cell_1_readvariableop_resourceOsequential_bidirectional_forward_gru_while_gru_cell_1_readvariableop_resource_0"s
3sequential_bidirectional_forward_gru_while_identity<sequential/bidirectional/forward_gru/while/Identity:output:0"w
5sequential_bidirectional_forward_gru_while_identity_1>sequential/bidirectional/forward_gru/while/Identity_1:output:0"w
5sequential_bidirectional_forward_gru_while_identity_2>sequential/bidirectional/forward_gru/while/Identity_2:output:0"w
5sequential_bidirectional_forward_gru_while_identity_3>sequential/bidirectional/forward_gru/while/Identity_3:output:0"w
5sequential_bidirectional_forward_gru_while_identity_4>sequential/bidirectional/forward_gru/while/Identity_4:output:0"Ä
_sequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_strided_slice_1asequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_strided_slice_1_0"¾
sequential_bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensorsequential_bidirectional_forward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Ksequential/bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOpKsequential/bidirectional/forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2
Msequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpMsequential/bidirectional/forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2
Dsequential/bidirectional/forward_gru/while/gru_cell_1/ReadVariableOpDsequential/bidirectional/forward_gru/while/gru_cell_1/ReadVariableOp: 
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
°	

-__inference_bidirectional_layer_call_fn_19881
inputs_0
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18089o
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
while_body_17390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_1_readvariableop_resource_0:C
1while_gru_cell_1_matmul_readvariableop_resource_0:E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_1_readvariableop_resource:A
/while_gru_cell_1_matmul_readvariableop_resource:C
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
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
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 
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



forward_gru_while_cond_206144
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_26
2forward_gru_while_less_forward_gru_strided_slice_1K
Gforward_gru_while_forward_gru_while_cond_20614___redundant_placeholder0K
Gforward_gru_while_forward_gru_while_cond_20614___redundant_placeholder1K
Gforward_gru_while_forward_gru_while_cond_20614___redundant_placeholder2K
Gforward_gru_while_forward_gru_while_cond_20614___redundant_placeholder3
forward_gru_while_identity

forward_gru/while/LessLessforward_gru_while_placeholder2forward_gru_while_less_forward_gru_strided_slice_1*
T0*
_output_shapes
: c
forward_gru/while/IdentityIdentityforward_gru/while/Less:z:0*
T0
*
_output_shapes
: "A
forward_gru_while_identity#forward_gru/while/Identity:output:0*(
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
³(
¤
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16823

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpf
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
¬
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
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
Ù
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namestates



forward_gru_while_cond_202964
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_26
2forward_gru_while_less_forward_gru_strided_slice_1K
Gforward_gru_while_forward_gru_while_cond_20296___redundant_placeholder0K
Gforward_gru_while_forward_gru_while_cond_20296___redundant_placeholder1K
Gforward_gru_while_forward_gru_while_cond_20296___redundant_placeholder2K
Gforward_gru_while_forward_gru_while_cond_20296___redundant_placeholder3
forward_gru_while_identity

forward_gru/while/LessLessforward_gru_while_placeholder2forward_gru_while_less_forward_gru_strided_slice_1*
T0*
_output_shapes
: c
forward_gru/while/IdentityIdentityforward_gru/while/Less:z:0*
T0
*
_output_shapes
: "A
forward_gru_while_identity#forward_gru/while/Identity:output:0*(
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
ÕX
Ó
F__inference_forward_gru_layer_call_and_return_conditional_losses_17485

inputs4
"gru_cell_1_readvariableop_resource:;
)gru_cell_1_matmul_readvariableop_resource:=
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
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
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
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
while_body_17390*
condR
while_cond_17389*8
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
 *    ·
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â<
ø
while_body_22351
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:C
1while_gru_cell_2_matmul_readvariableop_resource_0:E
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:A
/while_gru_cell_2_matmul_readvariableop_resource:C
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
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
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp: 
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
while_cond_21817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21817___redundant_placeholder03
/while_while_cond_21817___redundant_placeholder13
/while_while_cond_21817___redundant_placeholder23
/while_while_cond_21817___redundant_placeholder3
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
ÓÐ
ó
H__inference_bidirectional_layer_call_and_return_conditional_losses_18869

inputs@
.forward_gru_gru_cell_1_readvariableop_resource:G
5forward_gru_gru_cell_1_matmul_readvariableop_resource:I
7forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
A
/backward_gru_gru_cell_2_readvariableop_resource:H
6backward_gru_gru_cell_2_matmul_readvariableop_resource:J
8backward_gru_gru_cell_2_matmul_1_readvariableop_resource:

identity¢-backward_gru/gru_cell_2/MatMul/ReadVariableOp¢/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢&backward_gru/gru_cell_2/ReadVariableOp¢backward_gru/while¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢,forward_gru/gru_cell_1/MatMul/ReadVariableOp¢.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢%forward_gru/gru_cell_1/ReadVariableOp¢forward_gru/whileG
forward_gru/ShapeShapeinputs*
T0*
_output_shapes
:i
forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_sliceStridedSliceforward_gru/Shape:output:0(forward_gru/strided_slice/stack:output:0*forward_gru/strided_slice/stack_1:output:0*forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru/zeros/packedPack"forward_gru/strided_slice:output:0#forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru/zerosFill!forward_gru/zeros/packed:output:0 forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru/transpose	Transposeinputs#forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
forward_gru/Shape_1Shapeforward_gru/transpose:y:0*
T0*
_output_shapes
:k
!forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_slice_1StridedSliceforward_gru/Shape_1:output:0*forward_gru/strided_slice_1/stack:output:0,forward_gru/strided_slice_1/stack_1:output:0,forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
forward_gru/TensorArrayV2TensorListReserve0forward_gru/TensorArrayV2/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
3forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru/transpose:y:0Jforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
forward_gru/strided_slice_2StridedSliceforward_gru/transpose:y:0*forward_gru/strided_slice_2/stack:output:0,forward_gru/strided_slice_2/stack_1:output:0,forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
%forward_gru/gru_cell_1/ReadVariableOpReadVariableOp.forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0
forward_gru/gru_cell_1/unstackUnpack-forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¢
,forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
forward_gru/gru_cell_1/MatMulMatMul$forward_gru/strided_slice_2:output:04forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
forward_gru/gru_cell_1/BiasAddBiasAdd'forward_gru/gru_cell_1/MatMul:product:0'forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
forward_gru/gru_cell_1/splitSplit/forward_gru/gru_cell_1/split/split_dim:output:0'forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¦
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¯
forward_gru/gru_cell_1/MatMul_1MatMulforward_gru/zeros:output:06forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 forward_gru/gru_cell_1/BiasAdd_1BiasAdd)forward_gru/gru_cell_1/MatMul_1:product:0'forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿs
(forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
forward_gru/gru_cell_1/split_1SplitV)forward_gru/gru_cell_1/BiasAdd_1:output:0%forward_gru/gru_cell_1/Const:output:01forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¥
forward_gru/gru_cell_1/addAddV2%forward_gru/gru_cell_1/split:output:0'forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru/gru_cell_1/SigmoidSigmoidforward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
forward_gru/gru_cell_1/add_1AddV2%forward_gru/gru_cell_1/split:output:1'forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru/gru_cell_1/Sigmoid_1Sigmoid forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
forward_gru/gru_cell_1/mulMul$forward_gru/gru_cell_1/Sigmoid_1:y:0'forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_2AddV2%forward_gru/gru_cell_1/split:output:2forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
forward_gru/gru_cell_1/ReluRelu forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/mul_1Mul"forward_gru/gru_cell_1/Sigmoid:y:0forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
forward_gru/gru_cell_1/subSub%forward_gru/gru_cell_1/sub/x:output:0"forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
forward_gru/gru_cell_1/mul_2Mulforward_gru/gru_cell_1/sub:z:0)forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_3AddV2 forward_gru/gru_cell_1/mul_1:z:0 forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ü
forward_gru/TensorArrayV2_1TensorListReserve2forward_gru/TensorArrayV2_1/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
forward_gru/whileWhile'forward_gru/while/loop_counter:output:0-forward_gru/while/maximum_iterations:output:0forward_gru/time:output:0$forward_gru/TensorArrayV2_1:handle:0forward_gru/zeros:output:0$forward_gru/strided_slice_1:output:0Cforward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0.forward_gru_gru_cell_1_readvariableop_resource5forward_gru_gru_cell_1_matmul_readvariableop_resource7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *(
body R
forward_gru_while_body_18615*(
cond R
forward_gru_while_cond_18614*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
<forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   æ
.forward_gru/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru/while:output:3Eforward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0t
!forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
forward_gru/strided_slice_3StridedSlice7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0*forward_gru/strided_slice_3/stack:output:0,forward_gru/strided_slice_3/stack_1:output:0,forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskq
forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
forward_gru/transpose_1	Transpose7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0%forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    H
backward_gru/ShapeShapeinputs*
T0*
_output_shapes
:j
 backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_sliceStridedSlicebackward_gru/Shape:output:0)backward_gru/strided_slice/stack:output:0+backward_gru/strided_slice/stack_1:output:0+backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

backward_gru/zeros/packedPack#backward_gru/strided_slice:output:0$backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru/zerosFill"backward_gru/zeros/packed:output:0!backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru/transpose	Transposeinputs$backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
backward_gru/Shape_1Shapebackward_gru/transpose:y:0*
T0*
_output_shapes
:l
"backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_slice_1StridedSlicebackward_gru/Shape_1:output:0+backward_gru/strided_slice_1/stack:output:0-backward_gru/strided_slice_1/stack_1:output:0-backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
backward_gru/TensorArrayV2TensorListReserve1backward_gru/TensorArrayV2/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_gru/ReverseV2	ReverseV2backward_gru/transpose:y:0$backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
4backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbackward_gru/ReverseV2:output:0Kbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
backward_gru/strided_slice_2StridedSlicebackward_gru/transpose:y:0+backward_gru/strided_slice_2/stack:output:0-backward_gru/strided_slice_2/stack_1:output:0-backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
&backward_gru/gru_cell_2/ReadVariableOpReadVariableOp/backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0
backward_gru/gru_cell_2/unstackUnpack.backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¤
-backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
backward_gru/gru_cell_2/MatMulMatMul%backward_gru/strided_slice_2:output:05backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
backward_gru/gru_cell_2/BiasAddBiasAdd(backward_gru/gru_cell_2/MatMul:product:0(backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
backward_gru/gru_cell_2/splitSplit0backward_gru/gru_cell_2/split/split_dim:output:0(backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0²
 backward_gru/gru_cell_2/MatMul_1MatMulbackward_gru/zeros:output:07backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!backward_gru/gru_cell_2/BiasAdd_1BiasAdd*backward_gru/gru_cell_2/MatMul_1:product:0(backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿt
)backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
backward_gru/gru_cell_2/split_1SplitV*backward_gru/gru_cell_2/BiasAdd_1:output:0&backward_gru/gru_cell_2/Const:output:02backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
backward_gru/gru_cell_2/addAddV2&backward_gru/gru_cell_2/split:output:0(backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru/gru_cell_2/SigmoidSigmoidbackward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
backward_gru/gru_cell_2/add_1AddV2&backward_gru/gru_cell_2/split:output:1(backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru/gru_cell_2/Sigmoid_1Sigmoid!backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
backward_gru/gru_cell_2/mulMul%backward_gru/gru_cell_2/Sigmoid_1:y:0(backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
backward_gru/gru_cell_2/add_2AddV2&backward_gru/gru_cell_2/split:output:2backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
backward_gru/gru_cell_2/ReluRelu!backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/mul_1Mul#backward_gru/gru_cell_2/Sigmoid:y:0backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
backward_gru/gru_cell_2/subSub&backward_gru/gru_cell_2/sub/x:output:0#backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
backward_gru/gru_cell_2/mul_2Mulbackward_gru/gru_cell_2/sub:z:0*backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/add_3AddV2!backward_gru/gru_cell_2/mul_1:z:0!backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
*backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ß
backward_gru/TensorArrayV2_1TensorListReserve3backward_gru/TensorArrayV2_1/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
backward_gru/whileWhile(backward_gru/while/loop_counter:output:0.backward_gru/while/maximum_iterations:output:0backward_gru/time:output:0%backward_gru/TensorArrayV2_1:handle:0backward_gru/zeros:output:0%backward_gru/strided_slice_1:output:0Dbackward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/backward_gru_gru_cell_2_readvariableop_resource6backward_gru_gru_cell_2_matmul_readvariableop_resource8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *)
body!R
backward_gru_while_body_18766*)
cond!R
backward_gru_while_cond_18765*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
=backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   é
/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru/while:output:3Fbackward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
"backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
backward_gru/strided_slice_3StridedSlice8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0+backward_gru/strided_slice_3/stack:output:0-backward_gru/strided_slice_3/stack_1:output:0-backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskr
backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
backward_gru/transpose_1	Transpose8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0&backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2$forward_gru/strided_slice_3:output:0%backward_gru/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Å
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp.^backward_gru/gru_cell_2/MatMul/ReadVariableOp0^backward_gru/gru_cell_2/MatMul_1/ReadVariableOp'^backward_gru/gru_cell_2/ReadVariableOp^backward_gru/whileO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp-^forward_gru/gru_cell_1/MatMul/ReadVariableOp/^forward_gru/gru_cell_1/MatMul_1/ReadVariableOp&^forward_gru/gru_cell_1/ReadVariableOp^forward_gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_gru/gru_cell_2/MatMul/ReadVariableOp-backward_gru/gru_cell_2/MatMul/ReadVariableOp2b
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2P
&backward_gru/gru_cell_2/ReadVariableOp&backward_gru/gru_cell_2/ReadVariableOp2(
backward_gru/whilebackward_gru/while2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2\
,forward_gru/gru_cell_1/MatMul/ReadVariableOp,forward_gru/gru_cell_1/MatMul/ReadVariableOp2`
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2N
%forward_gru/gru_cell_1/ReadVariableOp%forward_gru/gru_cell_1/ReadVariableOp2&
forward_gru/whileforward_gru/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
?
Î
F__inference_forward_gru_layer_call_and_return_conditional_losses_16751

inputs"
gru_cell_1_16669:"
gru_cell_1_16671:"
gru_cell_1_16673:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_1/StatefulPartitionedCall¢while;
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
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_16669gru_cell_1_16671gru_cell_1_16673*
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16668n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_16669gru_cell_1_16671gru_cell_1_16673*
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
while_body_16681*
condR
while_cond_16680*8
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
 *    
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_1_16671*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»(
¦
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_22686

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpf
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
¬
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
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
Ù
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:O K
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
¹<
ø
while_body_22029
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:C
1while_gru_cell_2_matmul_readvariableop_resource_0:E
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:A
/while_gru_cell_2_matmul_readvariableop_resource:C
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
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
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp: 
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
Ê(
§
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_22821

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpf
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
­
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
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
Ú
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:O K
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
Â<
ø
while_body_21659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_1_readvariableop_resource_0:C
1while_gru_cell_1_matmul_readvariableop_resource_0:E
3while_gru_cell_1_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_1_readvariableop_resource:A
/while_gru_cell_1_matmul_readvariableop_resource:C
1while_gru_cell_1_matmul_1_readvariableop_resource:
¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0while/gru_cell_1/Const:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
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
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 
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
ð*
Ñ
E__inference_sequential_layer_call_and_return_conditional_losses_18950

inputs%
bidirectional_18914:%
bidirectional_18916:%
bidirectional_18918:
%
bidirectional_18920:%
bidirectional_18922:%
bidirectional_18924:

dense_18927:
dense_18929:
dense_1_18932:
dense_1_18934:
identity¢%bidirectional/StatefulPartitionedCall¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÝ
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallinputsbidirectional_18914bidirectional_18916bidirectional_18918bidirectional_18920bidirectional_18922bidirectional_18924*
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18869
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_18927dense_18929*
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
GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18453
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_18932dense_1_18934*
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
B__inference_dense_1_layer_call_and_return_conditional_losses_18470¡
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_18916*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_18922*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp&^bidirectional/StatefulPartitionedCallO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÁK
à

forward_gru_while_body_206154
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_23
/forward_gru_while_forward_gru_strided_slice_1_0o
kforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0H
6forward_gru_while_gru_cell_1_readvariableop_resource_0:O
=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:Q
?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:

forward_gru_while_identity 
forward_gru_while_identity_1 
forward_gru_while_identity_2 
forward_gru_while_identity_3 
forward_gru_while_identity_41
-forward_gru_while_forward_gru_strided_slice_1m
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorF
4forward_gru_while_gru_cell_1_readvariableop_resource:M
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource:O
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢+forward_gru/while/gru_cell_1/ReadVariableOp
Cforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   â
5forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0forward_gru_while_placeholderLforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¢
+forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOp6forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
$forward_gru/while/gru_cell_1/unstackUnpack3forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num°
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
#forward_gru/while/gru_cell_1/MatMulMatMul<forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0:forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$forward_gru/while/gru_cell_1/BiasAddBiasAdd-forward_gru/while/gru_cell_1/MatMul:product:0-forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
"forward_gru/while/gru_cell_1/splitSplit5forward_gru/while/gru_cell_1/split/split_dim:output:0-forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split´
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0À
%forward_gru/while/gru_cell_1/MatMul_1MatMulforward_gru_while_placeholder_2<forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
&forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd/forward_gru/while/gru_cell_1/MatMul_1:product:0-forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿy
.forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
$forward_gru/while/gru_cell_1/split_1SplitV/forward_gru/while/gru_cell_1/BiasAdd_1:output:0+forward_gru/while/gru_cell_1/Const:output:07forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split·
 forward_gru/while/gru_cell_1/addAddV2+forward_gru/while/gru_cell_1/split:output:0-forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$forward_gru/while/gru_cell_1/SigmoidSigmoid$forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
"forward_gru/while/gru_cell_1/add_1AddV2+forward_gru/while/gru_cell_1/split:output:1-forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid&forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
 forward_gru/while/gru_cell_1/mulMul*forward_gru/while/gru_cell_1/Sigmoid_1:y:0-forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
"forward_gru/while/gru_cell_1/add_2AddV2+forward_gru/while/gru_cell_1/split:output:2$forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!forward_gru/while/gru_cell_1/ReluRelu&forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
"forward_gru/while/gru_cell_1/mul_1Mul(forward_gru/while/gru_cell_1/Sigmoid:y:0forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
"forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
 forward_gru/while/gru_cell_1/subSub+forward_gru/while/gru_cell_1/sub/x:output:0(forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
"forward_gru/while/gru_cell_1/mul_2Mul$forward_gru/while/gru_cell_1/sub:z:0/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
"forward_gru/while/gru_cell_1/add_3AddV2&forward_gru/while/gru_cell_1/mul_1:z:0&forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
6forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemforward_gru_while_placeholder_1forward_gru_while_placeholder&forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒY
forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/addAddV2forward_gru_while_placeholder forward_gru/while/add/y:output:0*
T0*
_output_shapes
: [
forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/add_1AddV20forward_gru_while_forward_gru_while_loop_counter"forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: }
forward_gru/while/IdentityIdentityforward_gru/while/add_1:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: 
forward_gru/while/Identity_1Identity6forward_gru_while_forward_gru_while_maximum_iterations^forward_gru/while/NoOp*
T0*
_output_shapes
: }
forward_gru/while/Identity_2Identityforward_gru/while/add:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: ½
forward_gru/while/Identity_3IdentityFforward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
forward_gru/while/Identity_4Identity&forward_gru/while/gru_cell_1/add_3:z:0^forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
forward_gru/while/NoOpNoOp3^forward_gru/while/gru_cell_1/MatMul/ReadVariableOp5^forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp,^forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "`
-forward_gru_while_forward_gru_strided_slice_1/forward_gru_while_forward_gru_strided_slice_1_0"
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"|
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"n
4forward_gru_while_gru_cell_1_readvariableop_resource6forward_gru_while_gru_cell_1_readvariableop_resource_0"A
forward_gru_while_identity#forward_gru/while/Identity:output:0"E
forward_gru_while_identity_1%forward_gru/while/Identity_1:output:0"E
forward_gru_while_identity_2%forward_gru/while/Identity_2:output:0"E
forward_gru_while_identity_3%forward_gru/while/Identity_3:output:0"E
forward_gru_while_identity_4%forward_gru/while/Identity_4:output:0"Ø
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2h
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2l
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2Z
+forward_gru/while/gru_cell_1/ReadVariableOp+forward_gru/while/gru_cell_1/ReadVariableOp: 
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


ñ
@__inference_dense_layer_call_and_return_conditional_losses_21207

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
ß
¡
while_body_16681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_1_16703_0:*
while_gru_cell_1_16705_0:*
while_gru_cell_1_16707_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_1_16703:(
while_gru_cell_1_16705:(
while_gru_cell_1_16707:
¢(while/gru_cell_1/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_16703_0while_gru_cell_1_16705_0while_gru_cell_1_16707_0*
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16668Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_1_16703while_gru_cell_1_16703_0"2
while_gru_cell_1_16705while_gru_cell_1_16705_0"2
while_gru_cell_1_16707while_gru_cell_1_16707_0")
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
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 
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
while_cond_22350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22350___redundant_placeholder03
/while_while_cond_22350___redundant_placeholder13
/while_while_cond_22350___redundant_placeholder23
/while_while_cond_22350___redundant_placeholder3
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
ÓÐ
ó
H__inference_bidirectional_layer_call_and_return_conditional_losses_20869

inputs@
.forward_gru_gru_cell_1_readvariableop_resource:G
5forward_gru_gru_cell_1_matmul_readvariableop_resource:I
7forward_gru_gru_cell_1_matmul_1_readvariableop_resource:
A
/backward_gru_gru_cell_2_readvariableop_resource:H
6backward_gru_gru_cell_2_matmul_readvariableop_resource:J
8backward_gru_gru_cell_2_matmul_1_readvariableop_resource:

identity¢-backward_gru/gru_cell_2/MatMul/ReadVariableOp¢/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp¢&backward_gru/gru_cell_2/ReadVariableOp¢backward_gru/while¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢,forward_gru/gru_cell_1/MatMul/ReadVariableOp¢.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp¢%forward_gru/gru_cell_1/ReadVariableOp¢forward_gru/whileG
forward_gru/ShapeShapeinputs*
T0*
_output_shapes
:i
forward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!forward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!forward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_sliceStridedSliceforward_gru/Shape:output:0(forward_gru/strided_slice/stack:output:0*forward_gru/strided_slice/stack_1:output:0*forward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
forward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

forward_gru/zeros/packedPack"forward_gru/strided_slice:output:0#forward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
forward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
forward_gru/zerosFill!forward_gru/zeros/packed:output:0 forward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
forward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
forward_gru/transpose	Transposeinputs#forward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
forward_gru/Shape_1Shapeforward_gru/transpose:y:0*
T0*
_output_shapes
:k
!forward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
forward_gru/strided_slice_1StridedSliceforward_gru/Shape_1:output:0*forward_gru/strided_slice_1/stack:output:0,forward_gru/strided_slice_1/stack_1:output:0,forward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
'forward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿØ
forward_gru/TensorArrayV2TensorListReserve0forward_gru/TensorArrayV2/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
3forward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_gru/transpose:y:0Jforward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
!forward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#forward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
forward_gru/strided_slice_2StridedSliceforward_gru/transpose:y:0*forward_gru/strided_slice_2/stack:output:0,forward_gru/strided_slice_2/stack_1:output:0,forward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
%forward_gru/gru_cell_1/ReadVariableOpReadVariableOp.forward_gru_gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0
forward_gru/gru_cell_1/unstackUnpack-forward_gru/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¢
,forward_gru/gru_cell_1/MatMul/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
forward_gru/gru_cell_1/MatMulMatMul$forward_gru/strided_slice_2:output:04forward_gru/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
forward_gru/gru_cell_1/BiasAddBiasAdd'forward_gru/gru_cell_1/MatMul:product:0'forward_gru/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
&forward_gru/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿè
forward_gru/gru_cell_1/splitSplit/forward_gru/gru_cell_1/split/split_dim:output:0'forward_gru/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¦
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0¯
forward_gru/gru_cell_1/MatMul_1MatMulforward_gru/zeros:output:06forward_gru/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
 forward_gru/gru_cell_1/BiasAdd_1BiasAdd)forward_gru/gru_cell_1/MatMul_1:product:0'forward_gru/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
forward_gru/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿs
(forward_gru/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
forward_gru/gru_cell_1/split_1SplitV)forward_gru/gru_cell_1/BiasAdd_1:output:0%forward_gru/gru_cell_1/Const:output:01forward_gru/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¥
forward_gru/gru_cell_1/addAddV2%forward_gru/gru_cell_1/split:output:0'forward_gru/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
forward_gru/gru_cell_1/SigmoidSigmoidforward_gru/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
forward_gru/gru_cell_1/add_1AddV2%forward_gru/gru_cell_1/split:output:1'forward_gru/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 forward_gru/gru_cell_1/Sigmoid_1Sigmoid forward_gru/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
forward_gru/gru_cell_1/mulMul$forward_gru/gru_cell_1/Sigmoid_1:y:0'forward_gru/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_2AddV2%forward_gru/gru_cell_1/split:output:2forward_gru/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
forward_gru/gru_cell_1/ReluRelu forward_gru/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/mul_1Mul"forward_gru/gru_cell_1/Sigmoid:y:0forward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
forward_gru/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
forward_gru/gru_cell_1/subSub%forward_gru/gru_cell_1/sub/x:output:0"forward_gru/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
forward_gru/gru_cell_1/mul_2Mulforward_gru/gru_cell_1/sub:z:0)forward_gru/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

forward_gru/gru_cell_1/add_3AddV2 forward_gru/gru_cell_1/mul_1:z:0 forward_gru/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)forward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   Ü
forward_gru/TensorArrayV2_1TensorListReserve2forward_gru/TensorArrayV2_1/element_shape:output:0$forward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒR
forward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$forward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ`
forward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Õ
forward_gru/whileWhile'forward_gru/while/loop_counter:output:0-forward_gru/while/maximum_iterations:output:0forward_gru/time:output:0$forward_gru/TensorArrayV2_1:handle:0forward_gru/zeros:output:0$forward_gru/strided_slice_1:output:0Cforward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0.forward_gru_gru_cell_1_readvariableop_resource5forward_gru_gru_cell_1_matmul_readvariableop_resource7forward_gru_gru_cell_1_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *(
body R
forward_gru_while_body_20615*(
cond R
forward_gru_while_cond_20614*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
<forward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   æ
.forward_gru/TensorArrayV2Stack/TensorListStackTensorListStackforward_gru/while:output:3Eforward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0t
!forward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#forward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#forward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ã
forward_gru/strided_slice_3StridedSlice7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0*forward_gru/strided_slice_3/stack:output:0,forward_gru/strided_slice_3/stack_1:output:0,forward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskq
forward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
forward_gru/transpose_1	Transpose7forward_gru/TensorArrayV2Stack/TensorListStack:tensor:0%forward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
forward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    H
backward_gru/ShapeShapeinputs*
T0*
_output_shapes
:j
 backward_gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"backward_gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"backward_gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_sliceStridedSlicebackward_gru/Shape:output:0)backward_gru/strided_slice/stack:output:0+backward_gru/strided_slice/stack_1:output:0+backward_gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
backward_gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

backward_gru/zeros/packedPack#backward_gru/strided_slice:output:0$backward_gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
backward_gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
backward_gru/zerosFill"backward_gru/zeros/packed:output:0!backward_gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
backward_gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
backward_gru/transpose	Transposeinputs$backward_gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
backward_gru/Shape_1Shapebackward_gru/transpose:y:0*
T0*
_output_shapes
:l
"backward_gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
backward_gru/strided_slice_1StridedSlicebackward_gru/Shape_1:output:0+backward_gru/strided_slice_1/stack:output:0-backward_gru/strided_slice_1/stack_1:output:0-backward_gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(backward_gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
backward_gru/TensorArrayV2TensorListReserve1backward_gru/TensorArrayV2/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
backward_gru/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
backward_gru/ReverseV2	ReverseV2backward_gru/transpose:y:0$backward_gru/ReverseV2/axis:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
4backward_gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbackward_gru/ReverseV2:output:0Kbackward_gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"backward_gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$backward_gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
backward_gru/strided_slice_2StridedSlicebackward_gru/transpose:y:0+backward_gru/strided_slice_2/stack:output:0-backward_gru/strided_slice_2/stack_1:output:0-backward_gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
&backward_gru/gru_cell_2/ReadVariableOpReadVariableOp/backward_gru_gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0
backward_gru/gru_cell_2/unstackUnpack.backward_gru/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¤
-backward_gru/gru_cell_2/MatMul/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
backward_gru/gru_cell_2/MatMulMatMul%backward_gru/strided_slice_2:output:05backward_gru/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
backward_gru/gru_cell_2/BiasAddBiasAdd(backward_gru/gru_cell_2/MatMul:product:0(backward_gru/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'backward_gru/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
backward_gru/gru_cell_2/splitSplit0backward_gru/gru_cell_2/split/split_dim:output:0(backward_gru/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0²
 backward_gru/gru_cell_2/MatMul_1MatMulbackward_gru/zeros:output:07backward_gru/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!backward_gru/gru_cell_2/BiasAdd_1BiasAdd*backward_gru/gru_cell_2/MatMul_1:product:0(backward_gru/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
backward_gru/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿt
)backward_gru/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
backward_gru/gru_cell_2/split_1SplitV*backward_gru/gru_cell_2/BiasAdd_1:output:0&backward_gru/gru_cell_2/Const:output:02backward_gru/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¨
backward_gru/gru_cell_2/addAddV2&backward_gru/gru_cell_2/split:output:0(backward_gru/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
}
backward_gru/gru_cell_2/SigmoidSigmoidbackward_gru/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
backward_gru/gru_cell_2/add_1AddV2&backward_gru/gru_cell_2/split:output:1(backward_gru/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!backward_gru/gru_cell_2/Sigmoid_1Sigmoid!backward_gru/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
backward_gru/gru_cell_2/mulMul%backward_gru/gru_cell_2/Sigmoid_1:y:0(backward_gru/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
backward_gru/gru_cell_2/add_2AddV2&backward_gru/gru_cell_2/split:output:2backward_gru/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
y
backward_gru/gru_cell_2/ReluRelu!backward_gru/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/mul_1Mul#backward_gru/gru_cell_2/Sigmoid:y:0backward_gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
backward_gru/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
backward_gru/gru_cell_2/subSub&backward_gru/gru_cell_2/sub/x:output:0#backward_gru/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
backward_gru/gru_cell_2/mul_2Mulbackward_gru/gru_cell_2/sub:z:0*backward_gru/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

backward_gru/gru_cell_2/add_3AddV2!backward_gru/gru_cell_2/mul_1:z:0!backward_gru/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
*backward_gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ß
backward_gru/TensorArrayV2_1TensorListReserve3backward_gru/TensorArrayV2_1/element_shape:output:0%backward_gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
backward_gru/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%backward_gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
backward_gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
backward_gru/whileWhile(backward_gru/while/loop_counter:output:0.backward_gru/while/maximum_iterations:output:0backward_gru/time:output:0%backward_gru/TensorArrayV2_1:handle:0backward_gru/zeros:output:0%backward_gru/strided_slice_1:output:0Dbackward_gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/backward_gru_gru_cell_2_readvariableop_resource6backward_gru_gru_cell_2_matmul_readvariableop_resource8backward_gru_gru_cell_2_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *)
body!R
backward_gru_while_body_20766*)
cond!R
backward_gru_while_cond_20765*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : *
parallel_iterations 
=backward_gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   é
/backward_gru/TensorArrayV2Stack/TensorListStackTensorListStackbackward_gru/while:output:3Fbackward_gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
"backward_gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$backward_gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$backward_gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
backward_gru/strided_slice_3StridedSlice8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0+backward_gru/strided_slice_3/stack:output:0-backward_gru/strided_slice_3/stack_1:output:0-backward_gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskr
backward_gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
backward_gru/transpose_1	Transpose8backward_gru/TensorArrayV2Stack/TensorListStack:tensor:0&backward_gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
backward_gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
concatConcatV2$forward_gru/strided_slice_3:output:0%backward_gru/strided_slice_3:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5forward_gru_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: Å
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6backward_gru_gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
NoOpNoOp.^backward_gru/gru_cell_2/MatMul/ReadVariableOp0^backward_gru/gru_cell_2/MatMul_1/ReadVariableOp'^backward_gru/gru_cell_2/ReadVariableOp^backward_gru/whileO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp-^forward_gru/gru_cell_1/MatMul/ReadVariableOp/^forward_gru/gru_cell_1/MatMul_1/ReadVariableOp&^forward_gru/gru_cell_1/ReadVariableOp^forward_gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2^
-backward_gru/gru_cell_2/MatMul/ReadVariableOp-backward_gru/gru_cell_2/MatMul/ReadVariableOp2b
/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp/backward_gru/gru_cell_2/MatMul_1/ReadVariableOp2P
&backward_gru/gru_cell_2/ReadVariableOp&backward_gru/gru_cell_2/ReadVariableOp2(
backward_gru/whilebackward_gru/while2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2\
,forward_gru/gru_cell_1/MatMul/ReadVariableOp,forward_gru/gru_cell_1/MatMul/ReadVariableOp2`
.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp.forward_gru/gru_cell_1/MatMul_1/ReadVariableOp2N
%forward_gru/gru_cell_1/ReadVariableOp%forward_gru/gru_cell_1/ReadVariableOp2&
forward_gru/whileforward_gru/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¥
while_cond_17240
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_17240___redundant_placeholder03
/while_while_cond_17240___redundant_placeholder13
/while_while_cond_17240___redundant_placeholder23
/while_while_cond_17240___redundant_placeholder3
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
while_cond_22189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22189___redundant_placeholder03
/while_while_cond_22189___redundant_placeholder13
/while_while_cond_22189___redundant_placeholder23
/while_while_cond_22189___redundant_placeholder3
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
while_cond_17950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_17950___redundant_placeholder03
/while_while_cond_17950___redundant_placeholder13
/while_while_cond_17950___redundant_placeholder23
/while_while_cond_17950___redundant_placeholder3
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
²
è
__inference_loss_fn_0_22742h
Vbidirectional_forward_gru_gru_cell_1_kernel_regularizer_square_readvariableop_resource:
identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpä
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpVbidirectional_forward_gru_gru_cell_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
IdentityIdentity?bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp
Õ
¥
while_cond_17769
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_17769___redundant_placeholder03
/while_while_cond_17769___redundant_placeholder13
/while_while_cond_17769___redundant_placeholder23
/while_while_cond_17769___redundant_placeholder3
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
µ


backward_gru_while_cond_207656
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_28
4backward_gru_while_less_backward_gru_strided_slice_1M
Ibackward_gru_while_backward_gru_while_cond_20765___redundant_placeholder0M
Ibackward_gru_while_backward_gru_while_cond_20765___redundant_placeholder1M
Ibackward_gru_while_backward_gru_while_cond_20765___redundant_placeholder2M
Ibackward_gru_while_backward_gru_while_cond_20765___redundant_placeholder3
backward_gru_while_identity

backward_gru/while/LessLessbackward_gru_while_placeholder4backward_gru_while_less_backward_gru_strided_slice_1*
T0*
_output_shapes
: e
backward_gru/while/IdentityIdentitybackward_gru/while/Less:z:0*
T0
*
_output_shapes
: "C
backward_gru_while_identity$backward_gru/while/Identity:output:0*(
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
B__inference_dense_1_layer_call_and_return_conditional_losses_18470

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
Ê(
§
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_22866

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpf
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
­
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
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
Ú
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp*"
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
ReadVariableOpReadVariableOp2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:O K
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
Â<
ø
while_body_22512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
*while_gru_cell_2_readvariableop_resource_0:C
1while_gru_cell_2_matmul_readvariableop_resource_0:E
3while_gru_cell_2_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
(while_gru_cell_2_readvariableop_resource:A
/while_gru_cell_2_matmul_readvariableop_resource:C
1while_gru_cell_2_matmul_1_readvariableop_resource:
¢&while/gru_cell_2/MatMul/ReadVariableOp¢(while/gru_cell_2/MatMul_1/ReadVariableOp¢while/gru_cell_2/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¯
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0
while/gru_cell_2/ReadVariableOpReadVariableOp*while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_2/unstackUnpack'while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
&while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0µ
while/gru_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAddBiasAdd!while/gru_cell_2/MatMul:product:0!while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÖ
while/gru_cell_2/splitSplit)while/gru_cell_2/split/split_dim:output:0!while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
(while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
while/gru_cell_2/MatMul_1MatMulwhile_placeholder_20while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/BiasAdd_1BiasAdd#while/gru_cell_2/MatMul_1:product:0!while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿm
"while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_2/split_1SplitV#while/gru_cell_2/BiasAdd_1:output:0while/gru_cell_2/Const:output:0+while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
while/gru_cell_2/addAddV2while/gru_cell_2/split:output:0!while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
while/gru_cell_2/SigmoidSigmoidwhile/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_1AddV2while/gru_cell_2/split:output:1!while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
s
while/gru_cell_2/Sigmoid_1Sigmoidwhile/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mulMulwhile/gru_cell_2/Sigmoid_1:y:0!while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_2AddV2while/gru_cell_2/split:output:2while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
while/gru_cell_2/ReluReluwhile/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_1Mulwhile/gru_cell_2/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
[
while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_2/subSubwhile/gru_cell_2/sub/x:output:0while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/mul_2Mulwhile/gru_cell_2/sub:z:0#while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

while/gru_cell_2/add_3AddV2while/gru_cell_2/mul_1:z:0while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_2/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_2/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Â

while/NoOpNoOp'^while/gru_cell_2/MatMul/ReadVariableOp)^while/gru_cell_2/MatMul_1/ReadVariableOp ^while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_2_matmul_1_readvariableop_resource3while_gru_cell_2_matmul_1_readvariableop_resource_0"d
/while_gru_cell_2_matmul_readvariableop_resource1while_gru_cell_2_matmul_readvariableop_resource_0"V
(while_gru_cell_2_readvariableop_resource*while_gru_cell_2_readvariableop_resource_0")
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
&while/gru_cell_2/MatMul/ReadVariableOp&while/gru_cell_2/MatMul/ReadVariableOp2T
(while/gru_cell_2/MatMul_1/ReadVariableOp(while/gru_cell_2/MatMul_1/ReadVariableOp2B
while/gru_cell_2/ReadVariableOpwhile/gru_cell_2/ReadVariableOp: 
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
while_cond_21499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21499___redundant_placeholder03
/while_while_cond_21499___redundant_placeholder13
/while_while_cond_21499___redundant_placeholder23
/while_while_cond_21499___redundant_placeholder3
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
°	

-__inference_bidirectional_layer_call_fn_19864
inputs_0
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_17676o
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
¾

'__inference_dense_1_layer_call_fn_21216

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
B__inference_dense_1_layer_call_and_return_conditional_losses_18470o
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
Ï


*__inference_sequential_layer_call_fn_18512
bidirectional_input
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
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18489o
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
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namebidirectional_input
¢
¹
+__inference_forward_gru_layer_call_fn_21277

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallè
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
GPU 2J 8 *O
fJRH
F__inference_forward_gru_layer_call_and_return_conditional_losses_18046o
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
ß
¡
while_body_17045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_gru_cell_2_17067_0:*
while_gru_cell_2_17069_0:*
while_gru_cell_2_17071_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_gru_cell_2_17067:(
while_gru_cell_2_17069:(
while_gru_cell_2_17071:
¢(while/gru_cell_2/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0û
(while/gru_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_2_17067_0while_gru_cell_2_17069_0while_gru_cell_2_17071_0*
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17032Ú
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_2/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity1while/gru_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

while/NoOpNoOp)^while/gru_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "2
while_gru_cell_2_17067while_gru_cell_2_17067_0"2
while_gru_cell_2_17069while_gru_cell_2_17069_0"2
while_gru_cell_2_17071while_gru_cell_2_17071_0")
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
(while/gru_cell_2/StatefulPartitionedCall(while/gru_cell_2/StatefulPartitionedCall: 
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
ÁK
à

forward_gru_while_body_186154
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_23
/forward_gru_while_forward_gru_strided_slice_1_0o
kforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0H
6forward_gru_while_gru_cell_1_readvariableop_resource_0:O
=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:Q
?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:

forward_gru_while_identity 
forward_gru_while_identity_1 
forward_gru_while_identity_2 
forward_gru_while_identity_3 
forward_gru_while_identity_41
-forward_gru_while_forward_gru_strided_slice_1m
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorF
4forward_gru_while_gru_cell_1_readvariableop_resource:M
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource:O
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢+forward_gru/while/gru_cell_1/ReadVariableOp
Cforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   â
5forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0forward_gru_while_placeholderLforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¢
+forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOp6forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
$forward_gru/while/gru_cell_1/unstackUnpack3forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num°
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
#forward_gru/while/gru_cell_1/MatMulMatMul<forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0:forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$forward_gru/while/gru_cell_1/BiasAddBiasAdd-forward_gru/while/gru_cell_1/MatMul:product:0-forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
"forward_gru/while/gru_cell_1/splitSplit5forward_gru/while/gru_cell_1/split/split_dim:output:0-forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split´
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0À
%forward_gru/while/gru_cell_1/MatMul_1MatMulforward_gru_while_placeholder_2<forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
&forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd/forward_gru/while/gru_cell_1/MatMul_1:product:0-forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿy
.forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
$forward_gru/while/gru_cell_1/split_1SplitV/forward_gru/while/gru_cell_1/BiasAdd_1:output:0+forward_gru/while/gru_cell_1/Const:output:07forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split·
 forward_gru/while/gru_cell_1/addAddV2+forward_gru/while/gru_cell_1/split:output:0-forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$forward_gru/while/gru_cell_1/SigmoidSigmoid$forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
"forward_gru/while/gru_cell_1/add_1AddV2+forward_gru/while/gru_cell_1/split:output:1-forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid&forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
 forward_gru/while/gru_cell_1/mulMul*forward_gru/while/gru_cell_1/Sigmoid_1:y:0-forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
"forward_gru/while/gru_cell_1/add_2AddV2+forward_gru/while/gru_cell_1/split:output:2$forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!forward_gru/while/gru_cell_1/ReluRelu&forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
"forward_gru/while/gru_cell_1/mul_1Mul(forward_gru/while/gru_cell_1/Sigmoid:y:0forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
"forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
 forward_gru/while/gru_cell_1/subSub+forward_gru/while/gru_cell_1/sub/x:output:0(forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
"forward_gru/while/gru_cell_1/mul_2Mul$forward_gru/while/gru_cell_1/sub:z:0/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
"forward_gru/while/gru_cell_1/add_3AddV2&forward_gru/while/gru_cell_1/mul_1:z:0&forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
6forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemforward_gru_while_placeholder_1forward_gru_while_placeholder&forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒY
forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/addAddV2forward_gru_while_placeholder forward_gru/while/add/y:output:0*
T0*
_output_shapes
: [
forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/add_1AddV20forward_gru_while_forward_gru_while_loop_counter"forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: }
forward_gru/while/IdentityIdentityforward_gru/while/add_1:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: 
forward_gru/while/Identity_1Identity6forward_gru_while_forward_gru_while_maximum_iterations^forward_gru/while/NoOp*
T0*
_output_shapes
: }
forward_gru/while/Identity_2Identityforward_gru/while/add:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: ½
forward_gru/while/Identity_3IdentityFforward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
forward_gru/while/Identity_4Identity&forward_gru/while/gru_cell_1/add_3:z:0^forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
forward_gru/while/NoOpNoOp3^forward_gru/while/gru_cell_1/MatMul/ReadVariableOp5^forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp,^forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "`
-forward_gru_while_forward_gru_strided_slice_1/forward_gru_while_forward_gru_strided_slice_1_0"
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"|
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"n
4forward_gru_while_gru_cell_1_readvariableop_resource6forward_gru_while_gru_cell_1_readvariableop_resource_0"A
forward_gru_while_identity#forward_gru/while/Identity:output:0"E
forward_gru_while_identity_1%forward_gru/while/Identity_1:output:0"E
forward_gru_while_identity_2%forward_gru/while/Identity_2:output:0"E
forward_gru_while_identity_3%forward_gru/while/Identity_3:output:0"E
forward_gru_while_identity_4%forward_gru/while/Identity_4:output:0"Ø
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2h
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2l
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2Z
+forward_gru/while/gru_cell_1/ReadVariableOp+forward_gru/while/gru_cell_1/ReadVariableOp: 
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
+
Þ
E__inference_sequential_layer_call_and_return_conditional_losses_19076
bidirectional_input%
bidirectional_19040:%
bidirectional_19042:%
bidirectional_19044:
%
bidirectional_19046:%
bidirectional_19048:%
bidirectional_19050:

dense_19053:
dense_19055:
dense_1_19058:
dense_1_19060:
identity¢%bidirectional/StatefulPartitionedCall¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallê
%bidirectional/StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputbidirectional_19040bidirectional_19042bidirectional_19044bidirectional_19046bidirectional_19048bidirectional_19050*
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18869
dense/StatefulPartitionedCallStatefulPartitionedCall.bidirectional/StatefulPartitionedCall:output:0dense_19053dense_19055*
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
GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18453
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_19058dense_1_19060*
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
B__inference_dense_1_layer_call_and_return_conditional_losses_18470¡
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_19042*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbidirectional_19048*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp&^bidirectional/StatefulPartitionedCallO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2N
%bidirectional/StatefulPartitionedCall%bidirectional/StatefulPartitionedCall2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namebidirectional_input


ñ
@__inference_dense_layer_call_and_return_conditional_losses_18453

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
¤
º
,__inference_backward_gru_layer_call_fn_21952

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallé
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
GPU 2J 8 *P
fKRI
G__inference_backward_gru_layer_call_and_return_conditional_losses_17653o
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
µ


backward_gru_while_cond_187656
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_28
4backward_gru_while_less_backward_gru_strided_slice_1M
Ibackward_gru_while_backward_gru_while_cond_18765___redundant_placeholder0M
Ibackward_gru_while_backward_gru_while_cond_18765___redundant_placeholder1M
Ibackward_gru_while_backward_gru_while_cond_18765___redundant_placeholder2M
Ibackward_gru_while_backward_gru_while_cond_18765___redundant_placeholder3
backward_gru_while_identity

backward_gru/while/LessLessbackward_gru_while_placeholder4backward_gru_while_less_backward_gru_strided_slice_1*
T0*
_output_shapes
: e
backward_gru/while/IdentityIdentitybackward_gru/while/Less:z:0*
T0
*
_output_shapes
: "C
backward_gru_while_identity$backward_gru/while/Identity:output:0*(
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
£


#__inference_signature_wrapper_19835
bidirectional_input
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
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
 __inference__wrapped_model_16592o
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
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namebidirectional_input
ÍZ
Õ
G__inference_backward_gru_layer_call_and_return_conditional_losses_17865

inputs4
"gru_cell_2_readvariableop_resource:;
)gru_cell_2_matmul_readvariableop_resource:=
+gru_cell_2_matmul_1_readvariableop_resource:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_2/MatMul/ReadVariableOp¢"gru_cell_2/MatMul_1/ReadVariableOp¢gru_cell_2/ReadVariableOp¢while;
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
gru_cell_2/ReadVariableOpReadVariableOp"gru_cell_2_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_2/unstackUnpack!gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_2/MatMulMatMulstrided_slice_2:output:0(gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAddBiasAddgru_cell_2/MatMul:product:0gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_2/splitSplit#gru_cell_2/split/split_dim:output:0gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_2_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_2/MatMul_1MatMulzeros:output:0*gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_2/BiasAdd_1BiasAddgru_cell_2/MatMul_1:product:0gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_2/split_1SplitVgru_cell_2/BiasAdd_1:output:0gru_cell_2/Const:output:0%gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_2/addAddV2gru_cell_2/split:output:0gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_2/SigmoidSigmoidgru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_2/add_1AddV2gru_cell_2/split:output:1gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_2/Sigmoid_1Sigmoidgru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_2/mulMulgru_cell_2/Sigmoid_1:y:0gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_2/add_2AddV2gru_cell_2/split:output:2gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_2/ReluRelugru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_2/mul_1Mulgru_cell_2/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_2/subSubgru_cell_2/sub/x:output:0gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_2/mul_2Mulgru_cell_2/sub:z:0gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_2/add_3AddV2gru_cell_2/mul_1:z:0gru_cell_2/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_2_readvariableop_resource)gru_cell_2_matmul_readvariableop_resource+gru_cell_2_matmul_1_readvariableop_resource*
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
while_body_17770*
condR
while_cond_17769*8
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
 *    ¸
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_2/MatMul/ReadVariableOp#^gru_cell_2/MatMul_1/ReadVariableOp^gru_cell_2/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_2/MatMul/ReadVariableOp gru_cell_2/MatMul/ReadVariableOp2H
"gru_cell_2/MatMul_1/ReadVariableOp"gru_cell_2/MatMul_1/ReadVariableOp26
gru_cell_2/ReadVariableOpgru_cell_2/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

Ö
*__inference_gru_cell_1_layer_call_fn_22641

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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_16823o
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
¾
ä
5sequential_bidirectional_forward_gru_while_cond_16335f
bsequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_loop_counterl
hsequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_maximum_iterations:
6sequential_bidirectional_forward_gru_while_placeholder<
8sequential_bidirectional_forward_gru_while_placeholder_1<
8sequential_bidirectional_forward_gru_while_placeholder_2h
dsequential_bidirectional_forward_gru_while_less_sequential_bidirectional_forward_gru_strided_slice_1}
ysequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_cond_16335___redundant_placeholder0}
ysequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_cond_16335___redundant_placeholder1}
ysequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_cond_16335___redundant_placeholder2}
ysequential_bidirectional_forward_gru_while_sequential_bidirectional_forward_gru_while_cond_16335___redundant_placeholder37
3sequential_bidirectional_forward_gru_while_identity
ö
/sequential/bidirectional/forward_gru/while/LessLess6sequential_bidirectional_forward_gru_while_placeholderdsequential_bidirectional_forward_gru_while_less_sequential_bidirectional_forward_gru_strided_slice_1*
T0*
_output_shapes
: 
3sequential/bidirectional/forward_gru/while/IdentityIdentity3sequential/bidirectional/forward_gru/while/Less:z:0*
T0
*
_output_shapes
: "s
3sequential_bidirectional_forward_gru_while_identity<sequential/bidirectional/forward_gru/while/Identity:output:0*(
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
¤
º
,__inference_backward_gru_layer_call_fn_21963

inputs
unknown:
	unknown_0:
	unknown_1:

identity¢StatefulPartitionedCallé
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
GPU 2J 8 *P
fKRI
G__inference_backward_gru_layer_call_and_return_conditional_losses_17865o
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
¹X
Õ
F__inference_forward_gru_layer_call_and_return_conditional_losses_21595
inputs_04
"gru_cell_1_readvariableop_resource:;
)gru_cell_1_matmul_readvariableop_resource:=
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while=
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
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
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
while_body_21500*
condR
while_cond_21499*8
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
 *    ·
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Õ
¥
while_cond_21658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_21658___redundant_placeholder03
/while_while_cond_21658___redundant_placeholder13
/while_while_cond_21658___redundant_placeholder23
/while_while_cond_21658___redundant_placeholder3
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
	

-__inference_bidirectional_layer_call_fn_19898

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:

identity¢StatefulPartitionedCall
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
GPU 2J 8 *Q
fLRJ
H__inference_bidirectional_layer_call_and_return_conditional_losses_18428o
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
Ï


*__inference_sequential_layer_call_fn_18998
bidirectional_input
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
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallbidirectional_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18950o
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
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namebidirectional_input
A
Ð
G__inference_backward_gru_layer_call_and_return_conditional_losses_17311

inputs"
gru_cell_2_17229:"
gru_cell_2_17231:"
gru_cell_2_17233:

identity¢Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp¢"gru_cell_2/StatefulPartitionedCall¢while;
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
"gru_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_2_17229gru_cell_2_17231gru_cell_2_17233*
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_17187n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_2_17229gru_cell_2_17231gru_cell_2_17233*
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
while_body_17241*
condR
while_cond_17240*8
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
 *    
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgru_cell_2_17231*
_output_shapes

:*
dtype0Ê
?bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SquareSquareVbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ò
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/SumSumCbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square:y:0Gbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
>bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ô
<bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mulMulGbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/mul/x:output:0Ebidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ä
NoOpNoOpO^bidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp#^gru_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2 
Nbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOpNbidirectional/backward_gru/gru_cell_2/kernel/Regularizer/Square/ReadVariableOp2H
"gru_cell_2/StatefulPartitionedCall"gru_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÕX
Ó
F__inference_forward_gru_layer_call_and_return_conditional_losses_18046

inputs4
"gru_cell_1_readvariableop_resource:;
)gru_cell_1_matmul_readvariableop_resource:=
+gru_cell_1_matmul_1_readvariableop_resource:

identity¢Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢while;
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
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:*
dtype0u
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÄ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:
*
dtype0
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿg
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿò
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
|
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
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
while_body_17951*
condR
while_cond_17950*8
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
 *    ·
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0È
>bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SquareSquareUbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ï
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/SumSumBbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square:y:0Fbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 
=bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Q8ñ
;bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mulMulFbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/mul/x:output:0Dbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOpN^bidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2
Mbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOpMbidirectional/forward_gru/gru_cell_1/kernel/Regularizer/Square/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ãL
þ

backward_gru_while_body_187666
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_25
1backward_gru_while_backward_gru_strided_slice_1_0q
mbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0I
7backward_gru_while_gru_cell_2_readvariableop_resource_0:P
>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:R
@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

backward_gru_while_identity!
backward_gru_while_identity_1!
backward_gru_while_identity_2!
backward_gru_while_identity_3!
backward_gru_while_identity_43
/backward_gru_while_backward_gru_strided_slice_1o
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensorG
5backward_gru_while_gru_cell_2_readvariableop_resource:N
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource:P
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢,backward_gru/while/gru_cell_2/ReadVariableOp
Dbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ç
6backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0backward_gru_while_placeholderMbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¤
,backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOp7backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
%backward_gru/while/gru_cell_2/unstackUnpack4backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ü
$backward_gru/while/gru_cell_2/MatMulMatMul=backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0;backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
%backward_gru/while/gru_cell_2/BiasAddBiasAdd.backward_gru/while/gru_cell_2/MatMul:product:0.backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#backward_gru/while/gru_cell_2/splitSplit6backward_gru/while/gru_cell_2/split/split_dim:output:0.backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¶
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Ã
&backward_gru/while/gru_cell_2/MatMul_1MatMul backward_gru_while_placeholder_2=backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd0backward_gru/while/gru_cell_2/MatMul_1:product:0.backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿz
/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%backward_gru/while/gru_cell_2/split_1SplitV0backward_gru/while/gru_cell_2/BiasAdd_1:output:0,backward_gru/while/gru_cell_2/Const:output:08backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
!backward_gru/while/gru_cell_2/addAddV2,backward_gru/while/gru_cell_2/split:output:0.backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%backward_gru/while/gru_cell_2/SigmoidSigmoid%backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
#backward_gru/while/gru_cell_2/add_1AddV2,backward_gru/while/gru_cell_2/split:output:1.backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid'backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
!backward_gru/while/gru_cell_2/mulMul+backward_gru/while/gru_cell_2/Sigmoid_1:y:0.backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
#backward_gru/while/gru_cell_2/add_2AddV2,backward_gru/while/gru_cell_2/split:output:2%backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"backward_gru/while/gru_cell_2/ReluRelu'backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
#backward_gru/while/gru_cell_2/mul_1Mul)backward_gru/while/gru_cell_2/Sigmoid:y:0 backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
#backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!backward_gru/while/gru_cell_2/subSub,backward_gru/while/gru_cell_2/sub/x:output:0)backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
#backward_gru/while/gru_cell_2/mul_2Mul%backward_gru/while/gru_cell_2/sub:z:00backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
#backward_gru/while/gru_cell_2/add_3AddV2'backward_gru/while/gru_cell_2/mul_1:z:0'backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
7backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem backward_gru_while_placeholder_1backward_gru_while_placeholder'backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒZ
backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/addAddV2backward_gru_while_placeholder!backward_gru/while/add/y:output:0*
T0*
_output_shapes
: \
backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/add_1AddV22backward_gru_while_backward_gru_while_loop_counter#backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru/while/IdentityIdentitybackward_gru/while/add_1:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_1Identity8backward_gru_while_backward_gru_while_maximum_iterations^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_2Identitybackward_gru/while/add:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: À
backward_gru/while/Identity_3IdentityGbackward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
backward_gru/while/Identity_4Identity'backward_gru/while/gru_cell_2/add_3:z:0^backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
backward_gru/while/NoOpNoOp4^backward_gru/while/gru_cell_2/MatMul/ReadVariableOp6^backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp-^backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/backward_gru_while_backward_gru_strided_slice_11backward_gru_while_backward_gru_strided_slice_1_0"
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"~
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"p
5backward_gru_while_gru_cell_2_readvariableop_resource7backward_gru_while_gru_cell_2_readvariableop_resource_0"C
backward_gru_while_identity$backward_gru/while/Identity:output:0"G
backward_gru_while_identity_1&backward_gru/while/Identity_1:output:0"G
backward_gru_while_identity_2&backward_gru/while/Identity_2:output:0"G
backward_gru_while_identity_3&backward_gru/while/Identity_3:output:0"G
backward_gru_while_identity_4&backward_gru/while/Identity_4:output:0"Ü
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensormbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2j
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2n
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2\
,backward_gru/while/gru_cell_2/ReadVariableOp,backward_gru/while/gru_cell_2/ReadVariableOp: 
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
ìL
þ

backward_gru_while_body_204486
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_25
1backward_gru_while_backward_gru_strided_slice_1_0q
mbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0I
7backward_gru_while_gru_cell_2_readvariableop_resource_0:P
>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:R
@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

backward_gru_while_identity!
backward_gru_while_identity_1!
backward_gru_while_identity_2!
backward_gru_while_identity_3!
backward_gru_while_identity_43
/backward_gru_while_backward_gru_strided_slice_1o
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensorG
5backward_gru_while_gru_cell_2_readvariableop_resource:N
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource:P
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢,backward_gru/while/gru_cell_2/ReadVariableOp
Dbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿð
6backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0backward_gru_while_placeholderMbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¤
,backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOp7backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
%backward_gru/while/gru_cell_2/unstackUnpack4backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ü
$backward_gru/while/gru_cell_2/MatMulMatMul=backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0;backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
%backward_gru/while/gru_cell_2/BiasAddBiasAdd.backward_gru/while/gru_cell_2/MatMul:product:0.backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#backward_gru/while/gru_cell_2/splitSplit6backward_gru/while/gru_cell_2/split/split_dim:output:0.backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¶
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Ã
&backward_gru/while/gru_cell_2/MatMul_1MatMul backward_gru_while_placeholder_2=backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd0backward_gru/while/gru_cell_2/MatMul_1:product:0.backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿz
/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%backward_gru/while/gru_cell_2/split_1SplitV0backward_gru/while/gru_cell_2/BiasAdd_1:output:0,backward_gru/while/gru_cell_2/Const:output:08backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
!backward_gru/while/gru_cell_2/addAddV2,backward_gru/while/gru_cell_2/split:output:0.backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%backward_gru/while/gru_cell_2/SigmoidSigmoid%backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
#backward_gru/while/gru_cell_2/add_1AddV2,backward_gru/while/gru_cell_2/split:output:1.backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid'backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
!backward_gru/while/gru_cell_2/mulMul+backward_gru/while/gru_cell_2/Sigmoid_1:y:0.backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
#backward_gru/while/gru_cell_2/add_2AddV2,backward_gru/while/gru_cell_2/split:output:2%backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"backward_gru/while/gru_cell_2/ReluRelu'backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
#backward_gru/while/gru_cell_2/mul_1Mul)backward_gru/while/gru_cell_2/Sigmoid:y:0 backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
#backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!backward_gru/while/gru_cell_2/subSub,backward_gru/while/gru_cell_2/sub/x:output:0)backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
#backward_gru/while/gru_cell_2/mul_2Mul%backward_gru/while/gru_cell_2/sub:z:00backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
#backward_gru/while/gru_cell_2/add_3AddV2'backward_gru/while/gru_cell_2/mul_1:z:0'backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
7backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem backward_gru_while_placeholder_1backward_gru_while_placeholder'backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒZ
backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/addAddV2backward_gru_while_placeholder!backward_gru/while/add/y:output:0*
T0*
_output_shapes
: \
backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/add_1AddV22backward_gru_while_backward_gru_while_loop_counter#backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru/while/IdentityIdentitybackward_gru/while/add_1:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_1Identity8backward_gru_while_backward_gru_while_maximum_iterations^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_2Identitybackward_gru/while/add:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: À
backward_gru/while/Identity_3IdentityGbackward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
backward_gru/while/Identity_4Identity'backward_gru/while/gru_cell_2/add_3:z:0^backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
backward_gru/while/NoOpNoOp4^backward_gru/while/gru_cell_2/MatMul/ReadVariableOp6^backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp-^backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/backward_gru_while_backward_gru_strided_slice_11backward_gru_while_backward_gru_strided_slice_1_0"
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"~
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"p
5backward_gru_while_gru_cell_2_readvariableop_resource7backward_gru_while_gru_cell_2_readvariableop_resource_0"C
backward_gru_while_identity$backward_gru/while/Identity:output:0"G
backward_gru_while_identity_1&backward_gru/while/Identity_1:output:0"G
backward_gru_while_identity_2&backward_gru/while/Identity_2:output:0"G
backward_gru_while_identity_3&backward_gru/while/Identity_3:output:0"G
backward_gru_while_identity_4&backward_gru/while/Identity_4:output:0"Ü
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensormbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2j
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2n
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2\
,backward_gru/while/gru_cell_2/ReadVariableOp,backward_gru/while/gru_cell_2/ReadVariableOp: 
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
ÊK
à

forward_gru_while_body_202974
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_23
/forward_gru_while_forward_gru_strided_slice_1_0o
kforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0H
6forward_gru_while_gru_cell_1_readvariableop_resource_0:O
=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0:Q
?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0:

forward_gru_while_identity 
forward_gru_while_identity_1 
forward_gru_while_identity_2 
forward_gru_while_identity_3 
forward_gru_while_identity_41
-forward_gru_while_forward_gru_strided_slice_1m
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorF
4forward_gru_while_gru_cell_1_readvariableop_resource:M
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource:O
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource:
¢2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp¢4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp¢+forward_gru/while/gru_cell_1/ReadVariableOp
Cforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿë
5forward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0forward_gru_while_placeholderLforward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¢
+forward_gru/while/gru_cell_1/ReadVariableOpReadVariableOp6forward_gru_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
$forward_gru/while/gru_cell_1/unstackUnpack3forward_gru/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num°
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ù
#forward_gru/while/gru_cell_1/MatMulMatMul<forward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0:forward_gru/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$forward_gru/while/gru_cell_1/BiasAddBiasAdd-forward_gru/while/gru_cell_1/MatMul:product:0-forward_gru/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,forward_gru/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
"forward_gru/while/gru_cell_1/splitSplit5forward_gru/while/gru_cell_1/split/split_dim:output:0-forward_gru/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split´
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0À
%forward_gru/while/gru_cell_1/MatMul_1MatMulforward_gru_while_placeholder_2<forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
&forward_gru/while/gru_cell_1/BiasAdd_1BiasAdd/forward_gru/while/gru_cell_1/MatMul_1:product:0-forward_gru/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"forward_gru/while/gru_cell_1/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿy
.forward_gru/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿº
$forward_gru/while/gru_cell_1/split_1SplitV/forward_gru/while/gru_cell_1/BiasAdd_1:output:0+forward_gru/while/gru_cell_1/Const:output:07forward_gru/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split·
 forward_gru/while/gru_cell_1/addAddV2+forward_gru/while/gru_cell_1/split:output:0-forward_gru/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

$forward_gru/while/gru_cell_1/SigmoidSigmoid$forward_gru/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
"forward_gru/while/gru_cell_1/add_1AddV2+forward_gru/while/gru_cell_1/split:output:1-forward_gru/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&forward_gru/while/gru_cell_1/Sigmoid_1Sigmoid&forward_gru/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
´
 forward_gru/while/gru_cell_1/mulMul*forward_gru/while/gru_cell_1/Sigmoid_1:y:0-forward_gru/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
"forward_gru/while/gru_cell_1/add_2AddV2+forward_gru/while/gru_cell_1/split:output:2$forward_gru/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!forward_gru/while/gru_cell_1/ReluRelu&forward_gru/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
"forward_gru/while/gru_cell_1/mul_1Mul(forward_gru/while/gru_cell_1/Sigmoid:y:0forward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
"forward_gru/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
 forward_gru/while/gru_cell_1/subSub+forward_gru/while/gru_cell_1/sub/x:output:0(forward_gru/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²
"forward_gru/while/gru_cell_1/mul_2Mul$forward_gru/while/gru_cell_1/sub:z:0/forward_gru/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
­
"forward_gru/while/gru_cell_1/add_3AddV2&forward_gru/while/gru_cell_1/mul_1:z:0&forward_gru/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ó
6forward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemforward_gru_while_placeholder_1forward_gru_while_placeholder&forward_gru/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒY
forward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/addAddV2forward_gru_while_placeholder forward_gru/while/add/y:output:0*
T0*
_output_shapes
: [
forward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
forward_gru/while/add_1AddV20forward_gru_while_forward_gru_while_loop_counter"forward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: }
forward_gru/while/IdentityIdentityforward_gru/while/add_1:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: 
forward_gru/while/Identity_1Identity6forward_gru_while_forward_gru_while_maximum_iterations^forward_gru/while/NoOp*
T0*
_output_shapes
: }
forward_gru/while/Identity_2Identityforward_gru/while/add:z:0^forward_gru/while/NoOp*
T0*
_output_shapes
: ½
forward_gru/while/Identity_3IdentityFforward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
forward_gru/while/Identity_4Identity&forward_gru/while/gru_cell_1/add_3:z:0^forward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ò
forward_gru/while/NoOpNoOp3^forward_gru/while/gru_cell_1/MatMul/ReadVariableOp5^forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp,^forward_gru/while/gru_cell_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "`
-forward_gru_while_forward_gru_strided_slice_1/forward_gru_while_forward_gru_strided_slice_1_0"
=forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource?forward_gru_while_gru_cell_1_matmul_1_readvariableop_resource_0"|
;forward_gru_while_gru_cell_1_matmul_readvariableop_resource=forward_gru_while_gru_cell_1_matmul_readvariableop_resource_0"n
4forward_gru_while_gru_cell_1_readvariableop_resource6forward_gru_while_gru_cell_1_readvariableop_resource_0"A
forward_gru_while_identity#forward_gru/while/Identity:output:0"E
forward_gru_while_identity_1%forward_gru/while/Identity_1:output:0"E
forward_gru_while_identity_2%forward_gru/while/Identity_2:output:0"E
forward_gru_while_identity_3%forward_gru/while/Identity_3:output:0"E
forward_gru_while_identity_4%forward_gru/while/Identity_4:output:0"Ø
iforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensorkforward_gru_while_tensorarrayv2read_tensorlistgetitem_forward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2h
2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2forward_gru/while/gru_cell_1/MatMul/ReadVariableOp2l
4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp4forward_gru/while/gru_cell_1/MatMul_1/ReadVariableOp2Z
+forward_gru/while/gru_cell_1/ReadVariableOp+forward_gru/while/gru_cell_1/ReadVariableOp: 
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
while_cond_16680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_16680___redundant_placeholder03
/while_while_cond_16680___redundant_placeholder13
/while_while_cond_16680___redundant_placeholder23
/while_while_cond_16680___redundant_placeholder3
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
ìL
þ

backward_gru_while_body_201306
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_25
1backward_gru_while_backward_gru_strided_slice_1_0q
mbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0I
7backward_gru_while_gru_cell_2_readvariableop_resource_0:P
>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:R
@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

backward_gru_while_identity!
backward_gru_while_identity_1!
backward_gru_while_identity_2!
backward_gru_while_identity_3!
backward_gru_while_identity_43
/backward_gru_while_backward_gru_strided_slice_1o
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensorG
5backward_gru_while_gru_cell_2_readvariableop_resource:N
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource:P
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢,backward_gru/while/gru_cell_2/ReadVariableOp
Dbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿð
6backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0backward_gru_while_placeholderMbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0¤
,backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOp7backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
%backward_gru/while/gru_cell_2/unstackUnpack4backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ü
$backward_gru/while/gru_cell_2/MatMulMatMul=backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0;backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
%backward_gru/while/gru_cell_2/BiasAddBiasAdd.backward_gru/while/gru_cell_2/MatMul:product:0.backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#backward_gru/while/gru_cell_2/splitSplit6backward_gru/while/gru_cell_2/split/split_dim:output:0.backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¶
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Ã
&backward_gru/while/gru_cell_2/MatMul_1MatMul backward_gru_while_placeholder_2=backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd0backward_gru/while/gru_cell_2/MatMul_1:product:0.backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿz
/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%backward_gru/while/gru_cell_2/split_1SplitV0backward_gru/while/gru_cell_2/BiasAdd_1:output:0,backward_gru/while/gru_cell_2/Const:output:08backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
!backward_gru/while/gru_cell_2/addAddV2,backward_gru/while/gru_cell_2/split:output:0.backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%backward_gru/while/gru_cell_2/SigmoidSigmoid%backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
#backward_gru/while/gru_cell_2/add_1AddV2,backward_gru/while/gru_cell_2/split:output:1.backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid'backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
!backward_gru/while/gru_cell_2/mulMul+backward_gru/while/gru_cell_2/Sigmoid_1:y:0.backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
#backward_gru/while/gru_cell_2/add_2AddV2,backward_gru/while/gru_cell_2/split:output:2%backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"backward_gru/while/gru_cell_2/ReluRelu'backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
#backward_gru/while/gru_cell_2/mul_1Mul)backward_gru/while/gru_cell_2/Sigmoid:y:0 backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
#backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!backward_gru/while/gru_cell_2/subSub,backward_gru/while/gru_cell_2/sub/x:output:0)backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
#backward_gru/while/gru_cell_2/mul_2Mul%backward_gru/while/gru_cell_2/sub:z:00backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
#backward_gru/while/gru_cell_2/add_3AddV2'backward_gru/while/gru_cell_2/mul_1:z:0'backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
7backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem backward_gru_while_placeholder_1backward_gru_while_placeholder'backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒZ
backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/addAddV2backward_gru_while_placeholder!backward_gru/while/add/y:output:0*
T0*
_output_shapes
: \
backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/add_1AddV22backward_gru_while_backward_gru_while_loop_counter#backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru/while/IdentityIdentitybackward_gru/while/add_1:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_1Identity8backward_gru_while_backward_gru_while_maximum_iterations^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_2Identitybackward_gru/while/add:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: À
backward_gru/while/Identity_3IdentityGbackward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
backward_gru/while/Identity_4Identity'backward_gru/while/gru_cell_2/add_3:z:0^backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
backward_gru/while/NoOpNoOp4^backward_gru/while/gru_cell_2/MatMul/ReadVariableOp6^backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp-^backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/backward_gru_while_backward_gru_strided_slice_11backward_gru_while_backward_gru_strided_slice_1_0"
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"~
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"p
5backward_gru_while_gru_cell_2_readvariableop_resource7backward_gru_while_gru_cell_2_readvariableop_resource_0"C
backward_gru_while_identity$backward_gru/while/Identity:output:0"G
backward_gru_while_identity_1&backward_gru/while/Identity_1:output:0"G
backward_gru_while_identity_2&backward_gru/while/Identity_2:output:0"G
backward_gru_while_identity_3&backward_gru/while/Identity_3:output:0"G
backward_gru_while_identity_4&backward_gru/while/Identity_4:output:0"Ü
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensormbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2j
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2n
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2\
,backward_gru/while/gru_cell_2/ReadVariableOp,backward_gru/while/gru_cell_2/ReadVariableOp: 
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
l
ð
6sequential_bidirectional_backward_gru_while_body_16487h
dsequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_loop_countern
jsequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_maximum_iterations;
7sequential_bidirectional_backward_gru_while_placeholder=
9sequential_bidirectional_backward_gru_while_placeholder_1=
9sequential_bidirectional_backward_gru_while_placeholder_2g
csequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_strided_slice_1_0¤
sequential_bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0b
Psequential_bidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0:i
Wsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:k
Ysequential_bidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:
8
4sequential_bidirectional_backward_gru_while_identity:
6sequential_bidirectional_backward_gru_while_identity_1:
6sequential_bidirectional_backward_gru_while_identity_2:
6sequential_bidirectional_backward_gru_while_identity_3:
6sequential_bidirectional_backward_gru_while_identity_4e
asequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_strided_slice_1¢
sequential_bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor`
Nsequential_bidirectional_backward_gru_while_gru_cell_2_readvariableop_resource:g
Usequential_bidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource:i
Wsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢Lsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢Nsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢Esequential/bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp®
]sequential/bidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   å
Osequential/bidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_07sequential_bidirectional_backward_gru_while_placeholderfsequential/bidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ö
Esequential/bidirectional/backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOpPsequential_bidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0Í
>sequential/bidirectional/backward_gru/while/gru_cell_2/unstackUnpackMsequential/bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numä
Lsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOpWsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0§
=sequential/bidirectional/backward_gru/while/gru_cell_2/MatMulMatMulVsequential/bidirectional/backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0Tsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential/bidirectional/backward_gru/while/gru_cell_2/BiasAddBiasAddGsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul:product:0Gsequential/bidirectional/backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential/bidirectional/backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
<sequential/bidirectional/backward_gru/while/gru_cell_2/splitSplitOsequential/bidirectional/backward_gru/while/gru_cell_2/split/split_dim:output:0Gsequential/bidirectional/backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitè
Nsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOpYsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0
?sequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1MatMul9sequential_bidirectional_backward_gru_while_placeholder_2Vsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/bidirectional/backward_gru/while/gru_cell_2/BiasAdd_1BiasAddIsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1:product:0Gsequential/bidirectional/backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<sequential/bidirectional/backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿ
Hsequential/bidirectional/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
>sequential/bidirectional/backward_gru/while/gru_cell_2/split_1SplitVIsequential/bidirectional/backward_gru/while/gru_cell_2/BiasAdd_1:output:0Esequential/bidirectional/backward_gru/while/gru_cell_2/Const:output:0Qsequential/bidirectional/backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split
:sequential/bidirectional/backward_gru/while/gru_cell_2/addAddV2Esequential/bidirectional/backward_gru/while/gru_cell_2/split:output:0Gsequential/bidirectional/backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
»
>sequential/bidirectional/backward_gru/while/gru_cell_2/SigmoidSigmoid>sequential/bidirectional/backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<sequential/bidirectional/backward_gru/while/gru_cell_2/add_1AddV2Esequential/bidirectional/backward_gru/while/gru_cell_2/split:output:1Gsequential/bidirectional/backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿
@sequential/bidirectional/backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid@sequential/bidirectional/backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

:sequential/bidirectional/backward_gru/while/gru_cell_2/mulMulDsequential/bidirectional/backward_gru/while/gru_cell_2/Sigmoid_1:y:0Gsequential/bidirectional/backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
þ
<sequential/bidirectional/backward_gru/while/gru_cell_2/add_2AddV2Esequential/bidirectional/backward_gru/while/gru_cell_2/split:output:2>sequential/bidirectional/backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
;sequential/bidirectional/backward_gru/while/gru_cell_2/ReluRelu@sequential/bidirectional/backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ô
<sequential/bidirectional/backward_gru/while/gru_cell_2/mul_1MulBsequential/bidirectional/backward_gru/while/gru_cell_2/Sigmoid:y:09sequential_bidirectional_backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<sequential/bidirectional/backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?þ
:sequential/bidirectional/backward_gru/while/gru_cell_2/subSubEsequential/bidirectional/backward_gru/while/gru_cell_2/sub/x:output:0Bsequential/bidirectional/backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

<sequential/bidirectional/backward_gru/while/gru_cell_2/mul_2Mul>sequential/bidirectional/backward_gru/while/gru_cell_2/sub:z:0Isequential/bidirectional/backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
<sequential/bidirectional/backward_gru/while/gru_cell_2/add_3AddV2@sequential/bidirectional/backward_gru/while/gru_cell_2/mul_1:z:0@sequential/bidirectional/backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Û
Psequential/bidirectional/backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem9sequential_bidirectional_backward_gru_while_placeholder_17sequential_bidirectional_backward_gru_while_placeholder@sequential/bidirectional/backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒs
1sequential/bidirectional/backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Î
/sequential/bidirectional/backward_gru/while/addAddV27sequential_bidirectional_backward_gru_while_placeholder:sequential/bidirectional/backward_gru/while/add/y:output:0*
T0*
_output_shapes
: u
3sequential/bidirectional/backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ÿ
1sequential/bidirectional/backward_gru/while/add_1AddV2dsequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_loop_counter<sequential/bidirectional/backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: Ë
4sequential/bidirectional/backward_gru/while/IdentityIdentity5sequential/bidirectional/backward_gru/while/add_1:z:01^sequential/bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: 
6sequential/bidirectional/backward_gru/while/Identity_1Identityjsequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_while_maximum_iterations1^sequential/bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: Ë
6sequential/bidirectional/backward_gru/while/Identity_2Identity3sequential/bidirectional/backward_gru/while/add:z:01^sequential/bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: 
6sequential/bidirectional/backward_gru/while/Identity_3Identity`sequential/bidirectional/backward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^sequential/bidirectional/backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒé
6sequential/bidirectional/backward_gru/while/Identity_4Identity@sequential/bidirectional/backward_gru/while/gru_cell_2/add_3:z:01^sequential/bidirectional/backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ú
0sequential/bidirectional/backward_gru/while/NoOpNoOpM^sequential/bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpO^sequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpF^sequential/bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "´
Wsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resourceYsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"°
Usequential_bidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resourceWsequential_bidirectional_backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"¢
Nsequential_bidirectional_backward_gru_while_gru_cell_2_readvariableop_resourcePsequential_bidirectional_backward_gru_while_gru_cell_2_readvariableop_resource_0"u
4sequential_bidirectional_backward_gru_while_identity=sequential/bidirectional/backward_gru/while/Identity:output:0"y
6sequential_bidirectional_backward_gru_while_identity_1?sequential/bidirectional/backward_gru/while/Identity_1:output:0"y
6sequential_bidirectional_backward_gru_while_identity_2?sequential/bidirectional/backward_gru/while/Identity_2:output:0"y
6sequential_bidirectional_backward_gru_while_identity_3?sequential/bidirectional/backward_gru/while/Identity_3:output:0"y
6sequential_bidirectional_backward_gru_while_identity_4?sequential/bidirectional/backward_gru/while/Identity_4:output:0"È
asequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_strided_slice_1csequential_bidirectional_backward_gru_while_sequential_bidirectional_backward_gru_strided_slice_1_0"Â
sequential_bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensorsequential_bidirectional_backward_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_bidirectional_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2
Lsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOpLsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2 
Nsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpNsequential/bidirectional/backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2
Esequential/bidirectional/backward_gru/while/gru_cell_2/ReadVariableOpEsequential/bidirectional/backward_gru/while/gru_cell_2/ReadVariableOp: 
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
while_cond_17557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_17557___redundant_placeholder03
/while_while_cond_17557___redundant_placeholder13
/while_while_cond_17557___redundant_placeholder23
/while_while_cond_17557___redundant_placeholder3
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
°
¦
+bidirectional_backward_gru_while_cond_19690R
Nbidirectional_backward_gru_while_bidirectional_backward_gru_while_loop_counterX
Tbidirectional_backward_gru_while_bidirectional_backward_gru_while_maximum_iterations0
,bidirectional_backward_gru_while_placeholder2
.bidirectional_backward_gru_while_placeholder_12
.bidirectional_backward_gru_while_placeholder_2T
Pbidirectional_backward_gru_while_less_bidirectional_backward_gru_strided_slice_1i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19690___redundant_placeholder0i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19690___redundant_placeholder1i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19690___redundant_placeholder2i
ebidirectional_backward_gru_while_bidirectional_backward_gru_while_cond_19690___redundant_placeholder3-
)bidirectional_backward_gru_while_identity
Î
%bidirectional/backward_gru/while/LessLess,bidirectional_backward_gru_while_placeholderPbidirectional_backward_gru_while_less_bidirectional_backward_gru_strided_slice_1*
T0*
_output_shapes
: 
)bidirectional/backward_gru/while/IdentityIdentity)bidirectional/backward_gru/while/Less:z:0*
T0
*
_output_shapes
: "_
)bidirectional_backward_gru_while_identity2bidirectional/backward_gru/while/Identity:output:0*(
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
ãL
þ

backward_gru_while_body_183256
2backward_gru_while_backward_gru_while_loop_counter<
8backward_gru_while_backward_gru_while_maximum_iterations"
backward_gru_while_placeholder$
 backward_gru_while_placeholder_1$
 backward_gru_while_placeholder_25
1backward_gru_while_backward_gru_strided_slice_1_0q
mbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0I
7backward_gru_while_gru_cell_2_readvariableop_resource_0:P
>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0:R
@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0:

backward_gru_while_identity!
backward_gru_while_identity_1!
backward_gru_while_identity_2!
backward_gru_while_identity_3!
backward_gru_while_identity_43
/backward_gru_while_backward_gru_strided_slice_1o
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensorG
5backward_gru_while_gru_cell_2_readvariableop_resource:N
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource:P
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource:
¢3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp¢5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp¢,backward_gru/while/gru_cell_2/ReadVariableOp
Dbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ç
6backward_gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0backward_gru_while_placeholderMbackward_gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¤
,backward_gru/while/gru_cell_2/ReadVariableOpReadVariableOp7backward_gru_while_gru_cell_2_readvariableop_resource_0*
_output_shapes

:*
dtype0
%backward_gru/while/gru_cell_2/unstackUnpack4backward_gru/while/gru_cell_2/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOpReadVariableOp>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0Ü
$backward_gru/while/gru_cell_2/MatMulMatMul=backward_gru/while/TensorArrayV2Read/TensorListGetItem:item:0;backward_gru/while/gru_cell_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
%backward_gru/while/gru_cell_2/BiasAddBiasAdd.backward_gru/while/gru_cell_2/MatMul:product:0.backward_gru/while/gru_cell_2/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-backward_gru/while/gru_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#backward_gru/while/gru_cell_2/splitSplit6backward_gru/while/gru_cell_2/split/split_dim:output:0.backward_gru/while/gru_cell_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_split¶
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOpReadVariableOp@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0*
_output_shapes

:
*
dtype0Ã
&backward_gru/while/gru_cell_2/MatMul_1MatMul backward_gru_while_placeholder_2=backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
'backward_gru/while/gru_cell_2/BiasAdd_1BiasAdd0backward_gru/while/gru_cell_2/MatMul_1:product:0.backward_gru/while/gru_cell_2/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#backward_gru/while/gru_cell_2/ConstConst*
_output_shapes
:*
dtype0*!
valueB"
   
   ÿÿÿÿz
/backward_gru/while/gru_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%backward_gru/while/gru_cell_2/split_1SplitV0backward_gru/while/gru_cell_2/BiasAdd_1:output:0,backward_gru/while/gru_cell_2/Const:output:08backward_gru/while/gru_cell_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ
*
	num_splitº
!backward_gru/while/gru_cell_2/addAddV2,backward_gru/while/gru_cell_2/split:output:0.backward_gru/while/gru_cell_2/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%backward_gru/while/gru_cell_2/SigmoidSigmoid%backward_gru/while/gru_cell_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
#backward_gru/while/gru_cell_2/add_1AddV2,backward_gru/while/gru_cell_2/split:output:1.backward_gru/while/gru_cell_2/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'backward_gru/while/gru_cell_2/Sigmoid_1Sigmoid'backward_gru/while/gru_cell_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
!backward_gru/while/gru_cell_2/mulMul+backward_gru/while/gru_cell_2/Sigmoid_1:y:0.backward_gru/while/gru_cell_2/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
³
#backward_gru/while/gru_cell_2/add_2AddV2,backward_gru/while/gru_cell_2/split:output:2%backward_gru/while/gru_cell_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"backward_gru/while/gru_cell_2/ReluRelu'backward_gru/while/gru_cell_2/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
#backward_gru/while/gru_cell_2/mul_1Mul)backward_gru/while/gru_cell_2/Sigmoid:y:0 backward_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
#backward_gru/while/gru_cell_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!backward_gru/while/gru_cell_2/subSub,backward_gru/while/gru_cell_2/sub/x:output:0)backward_gru/while/gru_cell_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ
#backward_gru/while/gru_cell_2/mul_2Mul%backward_gru/while/gru_cell_2/sub:z:00backward_gru/while/gru_cell_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
#backward_gru/while/gru_cell_2/add_3AddV2'backward_gru/while/gru_cell_2/mul_1:z:0'backward_gru/while/gru_cell_2/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
7backward_gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem backward_gru_while_placeholder_1backward_gru_while_placeholder'backward_gru/while/gru_cell_2/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒZ
backward_gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/addAddV2backward_gru_while_placeholder!backward_gru/while/add/y:output:0*
T0*
_output_shapes
: \
backward_gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
backward_gru/while/add_1AddV22backward_gru_while_backward_gru_while_loop_counter#backward_gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
backward_gru/while/IdentityIdentitybackward_gru/while/add_1:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_1Identity8backward_gru_while_backward_gru_while_maximum_iterations^backward_gru/while/NoOp*
T0*
_output_shapes
: 
backward_gru/while/Identity_2Identitybackward_gru/while/add:z:0^backward_gru/while/NoOp*
T0*
_output_shapes
: À
backward_gru/while/Identity_3IdentityGbackward_gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
backward_gru/while/Identity_4Identity'backward_gru/while/gru_cell_2/add_3:z:0^backward_gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
backward_gru/while/NoOpNoOp4^backward_gru/while/gru_cell_2/MatMul/ReadVariableOp6^backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp-^backward_gru/while/gru_cell_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/backward_gru_while_backward_gru_strided_slice_11backward_gru_while_backward_gru_strided_slice_1_0"
>backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource@backward_gru_while_gru_cell_2_matmul_1_readvariableop_resource_0"~
<backward_gru_while_gru_cell_2_matmul_readvariableop_resource>backward_gru_while_gru_cell_2_matmul_readvariableop_resource_0"p
5backward_gru_while_gru_cell_2_readvariableop_resource7backward_gru_while_gru_cell_2_readvariableop_resource_0"C
backward_gru_while_identity$backward_gru/while/Identity:output:0"G
backward_gru_while_identity_1&backward_gru/while/Identity_1:output:0"G
backward_gru_while_identity_2&backward_gru/while/Identity_2:output:0"G
backward_gru_while_identity_3&backward_gru/while/Identity_3:output:0"G
backward_gru_while_identity_4&backward_gru/while/Identity_4:output:0"Ü
kbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensormbackward_gru_while_tensorarrayv2read_tensorlistgetitem_backward_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ
: : : : : 2j
3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp3backward_gru/while/gru_cell_2/MatMul/ReadVariableOp2n
5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp5backward_gru/while/gru_cell_2/MatMul_1/ReadVariableOp2\
,backward_gru/while/gru_cell_2/ReadVariableOp,backward_gru/while/gru_cell_2/ReadVariableOp: 
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



forward_gru_while_cond_199784
0forward_gru_while_forward_gru_while_loop_counter:
6forward_gru_while_forward_gru_while_maximum_iterations!
forward_gru_while_placeholder#
forward_gru_while_placeholder_1#
forward_gru_while_placeholder_26
2forward_gru_while_less_forward_gru_strided_slice_1K
Gforward_gru_while_forward_gru_while_cond_19978___redundant_placeholder0K
Gforward_gru_while_forward_gru_while_cond_19978___redundant_placeholder1K
Gforward_gru_while_forward_gru_while_cond_19978___redundant_placeholder2K
Gforward_gru_while_forward_gru_while_cond_19978___redundant_placeholder3
forward_gru_while_identity

forward_gru/while/LessLessforward_gru_while_placeholder2forward_gru_while_less_forward_gru_strided_slice_1*
T0*
_output_shapes
: c
forward_gru/while/IdentityIdentityforward_gru/while/Less:z:0*
T0
*
_output_shapes
: "A
forward_gru_while_identity#forward_gru/while/Identity:output:0*(
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


*bidirectional_forward_gru_while_cond_19207P
Lbidirectional_forward_gru_while_bidirectional_forward_gru_while_loop_counterV
Rbidirectional_forward_gru_while_bidirectional_forward_gru_while_maximum_iterations/
+bidirectional_forward_gru_while_placeholder1
-bidirectional_forward_gru_while_placeholder_11
-bidirectional_forward_gru_while_placeholder_2R
Nbidirectional_forward_gru_while_less_bidirectional_forward_gru_strided_slice_1g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19207___redundant_placeholder0g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19207___redundant_placeholder1g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19207___redundant_placeholder2g
cbidirectional_forward_gru_while_bidirectional_forward_gru_while_cond_19207___redundant_placeholder3,
(bidirectional_forward_gru_while_identity
Ê
$bidirectional/forward_gru/while/LessLess+bidirectional_forward_gru_while_placeholderNbidirectional_forward_gru_while_less_bidirectional_forward_gru_strided_slice_1*
T0*
_output_shapes
: 
(bidirectional/forward_gru/while/IdentityIdentity(bidirectional/forward_gru/while/Less:z:0*
T0
*
_output_shapes
: "]
(bidirectional_forward_gru_while_identity1bidirectional/forward_gru/while/Identity:output:0*(
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

NoOp*Æ
serving_default²
W
bidirectional_input@
%serving_default_bidirectional_input:0ÿÿÿÿÿÿÿÿÿ;
dense_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:òÅ
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
ö2ó
*__inference_sequential_layer_call_fn_18512
*__inference_sequential_layer_call_fn_19119
*__inference_sequential_layer_call_fn_19144
*__inference_sequential_layer_call_fn_18998À
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
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_19476
E__inference_sequential_layer_call_and_return_conditional_losses_19808
E__inference_sequential_layer_call_and_return_conditional_losses_19037
E__inference_sequential_layer_call_and_return_conditional_losses_19076À
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
×BÔ
 __inference__wrapped_model_16592bidirectional_input"
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
¨2¥
-__inference_bidirectional_layer_call_fn_19864
-__inference_bidirectional_layer_call_fn_19881
-__inference_bidirectional_layer_call_fn_19898
-__inference_bidirectional_layer_call_fn_19915æ
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
2
H__inference_bidirectional_layer_call_and_return_conditional_losses_20233
H__inference_bidirectional_layer_call_and_return_conditional_losses_20551
H__inference_bidirectional_layer_call_and_return_conditional_losses_20869
H__inference_bidirectional_layer_call_and_return_conditional_losses_21187æ
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
:2dense/kernel
:2
dense/bias
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
Ï2Ì
%__inference_dense_layer_call_fn_21196¢
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
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_21207¢
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
 :2dense_1/kernel
:2dense_1/bias
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
'__inference_dense_1_layer_call_fn_21216¢
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
B__inference_dense_1_layer_call_and_return_conditional_losses_21227¢
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
=:;2+bidirectional/forward_gru/gru_cell_1/kernel
G:E
25bidirectional/forward_gru/gru_cell_1/recurrent_kernel
;:92)bidirectional/forward_gru/gru_cell_1/bias
>:<2,bidirectional/backward_gru/gru_cell_2/kernel
H:F
26bidirectional/backward_gru/gru_cell_2/recurrent_kernel
<::2*bidirectional/backward_gru/gru_cell_2/bias
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
ÖBÓ
#__inference_signature_wrapper_19835bidirectional_input"
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
2
+__inference_forward_gru_layer_call_fn_21244
+__inference_forward_gru_layer_call_fn_21255
+__inference_forward_gru_layer_call_fn_21266
+__inference_forward_gru_layer_call_fn_21277Õ
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
û2ø
F__inference_forward_gru_layer_call_and_return_conditional_losses_21436
F__inference_forward_gru_layer_call_and_return_conditional_losses_21595
F__inference_forward_gru_layer_call_and_return_conditional_losses_21754
F__inference_forward_gru_layer_call_and_return_conditional_losses_21913Õ
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
2
,__inference_backward_gru_layer_call_fn_21930
,__inference_backward_gru_layer_call_fn_21941
,__inference_backward_gru_layer_call_fn_21952
,__inference_backward_gru_layer_call_fn_21963Õ
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
ÿ2ü
G__inference_backward_gru_layer_call_and_return_conditional_losses_22124
G__inference_backward_gru_layer_call_and_return_conditional_losses_22285
G__inference_backward_gru_layer_call_and_return_conditional_losses_22446
G__inference_backward_gru_layer_call_and_return_conditional_losses_22607Õ
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
*__inference_gru_cell_1_layer_call_fn_22627
*__inference_gru_cell_1_layer_call_fn_22641¾
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_22686
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_22731¾
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
__inference_loss_fn_0_22742
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
*__inference_gru_cell_2_layer_call_fn_22762
*__inference_gru_cell_2_layer_call_fn_22776¾
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_22821
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_22866¾
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
__inference_loss_fn_1_22877
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
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
B:@22Adam/bidirectional/forward_gru/gru_cell_1/kernel/m
L:J
2<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/m
@:>20Adam/bidirectional/forward_gru/gru_cell_1/bias/m
C:A23Adam/bidirectional/backward_gru/gru_cell_2/kernel/m
M:K
2=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/m
A:?21Adam/bidirectional/backward_gru/gru_cell_2/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
B:@22Adam/bidirectional/forward_gru/gru_cell_1/kernel/v
L:J
2<Adam/bidirectional/forward_gru/gru_cell_1/recurrent_kernel/v
@:>20Adam/bidirectional/forward_gru/gru_cell_1/bias/v
C:A23Adam/bidirectional/backward_gru/gru_cell_2/kernel/v
M:K
2=Adam/bidirectional/backward_gru/gru_cell_2/recurrent_kernel/v
A:?21Adam/bidirectional/backward_gru/gru_cell_2/bias/v¦
 __inference__wrapped_model_16592
,*+/-.@¢=
6¢3
1.
bidirectional_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿÈ
G__inference_backward_gru_layer_call_and_return_conditional_losses_22124}/-.O¢L
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
 È
G__inference_backward_gru_layer_call_and_return_conditional_losses_22285}/-.O¢L
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
 Ê
G__inference_backward_gru_layer_call_and_return_conditional_losses_22446/-.Q¢N
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
 Ê
G__inference_backward_gru_layer_call_and_return_conditional_losses_22607/-.Q¢N
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
  
,__inference_backward_gru_layer_call_fn_21930p/-.O¢L
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
 
,__inference_backward_gru_layer_call_fn_21941p/-.O¢L
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
¢
,__inference_backward_gru_layer_call_fn_21952r/-.Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¢
,__inference_backward_gru_layer_call_fn_21963r/-.Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
Ú
H__inference_bidirectional_layer_call_and_return_conditional_losses_20233,*+/-.\¢Y
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
 Ú
H__inference_bidirectional_layer_call_and_return_conditional_losses_20551,*+/-.\¢Y
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
 À
H__inference_bidirectional_layer_call_and_return_conditional_losses_20869t,*+/-.C¢@
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
 À
H__inference_bidirectional_layer_call_and_return_conditional_losses_21187t,*+/-.C¢@
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
 ²
-__inference_bidirectional_layer_call_fn_19864,*+/-.\¢Y
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
ª "ÿÿÿÿÿÿÿÿÿ²
-__inference_bidirectional_layer_call_fn_19881,*+/-.\¢Y
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
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_bidirectional_layer_call_fn_19898g,*+/-.C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_bidirectional_layer_call_fn_19915g,*+/-.C¢@
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
B__inference_dense_1_layer_call_and_return_conditional_losses_21227\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_1_layer_call_fn_21216O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_21207\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_layer_call_fn_21196O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÇ
F__inference_forward_gru_layer_call_and_return_conditional_losses_21436},*+O¢L
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
 Ç
F__inference_forward_gru_layer_call_and_return_conditional_losses_21595},*+O¢L
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
 É
F__inference_forward_gru_layer_call_and_return_conditional_losses_21754,*+Q¢N
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
 É
F__inference_forward_gru_layer_call_and_return_conditional_losses_21913,*+Q¢N
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
 
+__inference_forward_gru_layer_call_fn_21244p,*+O¢L
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

+__inference_forward_gru_layer_call_fn_21255p,*+O¢L
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
¡
+__inference_forward_gru_layer_call_fn_21266r,*+Q¢N
G¢D
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¡
+__inference_forward_gru_layer_call_fn_21277r,*+Q¢N
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_22686·,*+\¢Y
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
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_22731·,*+\¢Y
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
*__inference_gru_cell_1_layer_call_fn_22627©,*+\¢Y
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
*__inference_gru_cell_1_layer_call_fn_22641©,*+\¢Y
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_22821·/-.\¢Y
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
E__inference_gru_cell_2_layer_call_and_return_conditional_losses_22866·/-.\¢Y
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
*__inference_gru_cell_2_layer_call_fn_22762©/-.\¢Y
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
*__inference_gru_cell_2_layer_call_fn_22776©/-.\¢Y
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
__inference_loss_fn_0_22742*¢

¢ 
ª " :
__inference_loss_fn_1_22877-¢

¢ 
ª " Æ
E__inference_sequential_layer_call_and_return_conditional_losses_19037}
,*+/-.H¢E
>¢;
1.
bidirectional_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
E__inference_sequential_layer_call_and_return_conditional_losses_19076}
,*+/-.H¢E
>¢;
1.
bidirectional_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
E__inference_sequential_layer_call_and_return_conditional_losses_19476p
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
 ¹
E__inference_sequential_layer_call_and_return_conditional_losses_19808p
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
 
*__inference_sequential_layer_call_fn_18512p
,*+/-.H¢E
>¢;
1.
bidirectional_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_18998p
,*+/-.H¢E
>¢;
1.
bidirectional_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_19119c
,*+/-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_19144c
,*+/-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
#__inference_signature_wrapper_19835
,*+/-.W¢T
¢ 
MªJ
H
bidirectional_input1.
bidirectional_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ