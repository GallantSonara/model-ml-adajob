��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
ConstConst*
_output_shapes
:d*
dtype0	*�
value�B�	d"�                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       
�
Const_1Const*
_output_shapes
:d*
dtype0*�
value�B�dB1B10B100B11B12B13B14B15B16B17B18B19B2B20B21B22B23B24B25B26B27B28B29B3B30B31B32B33B34B35B36B37B38B39B4B40B41B42B43B44B45B46B47B48B49B5B50B51B52B53B54B55B56B57B58B59B6B60B61B62B63B64B65B66B67B68B69B7B70B71B72B73B74B75B76B77B78B79B8B80B81B82B83B84B85B86B87B88B89B9B90B91B92B93B94B95B96B97B98B99
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name116*
value_dtype0	
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e *%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:e *
dtype0
p

candidatesVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_name
candidates
i
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes

: *
dtype0
n
identifiersVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameidentifiers
g
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes
:*
dtype0
r
serving_default_input_1Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
hash_tableConst_2embedding/embeddings
candidatesidentifiers*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_12874
�
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__initializer_13031
(
NoOpNoOp^StatefulPartitionedCall_1
�
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
query_model
	identifiers
	_identifiers


candidates

_candidates
query_with_exclusions

signatures*

0
	1

2*

0*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 

	capture_1* 
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
KE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE*
* 

$serving_default* 
TN
VARIABLE_VALUEembedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

	0

1*

0*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 
#
%	keras_api
&lookup_table* 
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

embeddings*

0*

0*
* 
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
6
2trace_0
3trace_1
4trace_2
5trace_3* 
6
6trace_0
7trace_1
8trace_2
9trace_3* 

	capture_1* 
* 
R
:_initializer
;_create_resource
<_initialize
=_destroy_resource* 

0*

0*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
* 

0
1*
* 
* 
* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 

	capture_1* 
* 

Etrace_0* 

Ftrace_0* 

Gtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
 
H	capture_1
I	capture_2* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding/embeddingsConst_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_13083
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_13102�
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_13002

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	2
 embedding_embedding_lookup_12996:e 
identity��embedding/embedding_lookup�1string_lookup/hash_table_Lookup/LookupTableFindV2�
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
embedding/embedding_lookupResourceGather embedding_embedding_lookup_12996string_lookup/Identity:output:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/12996*'
_output_shapes
:��������� *
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/12996*'
_output_shapes
:��������� �
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� }
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^embedding/embedding_lookup2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 28
embedding/embedding_lookupembedding/embedding_lookup2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_12662

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	!
embedding_12658:e 
identity��!embedding/StatefulPartitionedCall�1string_lookup/hash_table_Lookup/LookupTableFindV2�
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_12658*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12610y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12768
queries
sequential_12750
sequential_12752	"
sequential_12754:e 0
matmul_readvariableop_resource: 
gather_resource:

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_12750sequential_12752sequential_12754*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12640t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_12989

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	2
 embedding_embedding_lookup_12983:e 
identity��embedding/embedding_lookup�1string_lookup/hash_table_Lookup/LookupTableFindV2�
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
embedding/embedding_lookupResourceGather embedding_embedding_lookup_12983string_lookup/Identity:output:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/12983*'
_output_shapes
:��������� *
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/12983*'
_output_shapes
:��������� �
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� }
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^embedding/embedding_lookup2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 28
embedding/embedding_lookupembedding/embedding_lookup2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_12610

inputs	(
embedding_lookup_12604:e 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_12604inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12604*'
_output_shapes
:��������� *
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12604*'
_output_shapes
:��������� }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:��������� Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12954
queriesM
Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleN
Jsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	=
+sequential_embedding_embedding_lookup_12938:e 0
matmul_readvariableop_resource: 
gather_resource:

identity_1

identity_2��Gather�MatMul/ReadVariableOp�%sequential/embedding/embedding_lookup�<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2�
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handlequeriesJsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
!sequential/string_lookup/IdentityIdentityEsequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_12938*sequential/string_lookup/Identity:output:0*
Tindices0	*>
_class4
20loc:@sequential/embedding/embedding_lookup/12938*'
_output_shapes
:��������� *
dtype0�
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/12938*'
_output_shapes
:��������� �
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup=^sequential/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2|
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12931
queriesM
Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleN
Jsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	=
+sequential_embedding_embedding_lookup_12915:e 0
matmul_readvariableop_resource: 
gather_resource:

identity_1

identity_2��Gather�MatMul/ReadVariableOp�%sequential/embedding/embedding_lookup�<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2�
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Isequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handlequeriesJsequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
!sequential/string_lookup/IdentityIdentityEsequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
%sequential/embedding/embedding_lookupResourceGather+sequential_embedding_embedding_lookup_12915*sequential/string_lookup/Identity:output:0*
Tindices0	*>
_class4
20loc:@sequential/embedding/embedding_lookup/12915*'
_output_shapes
:��������� *
dtype0�
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@sequential/embedding/embedding_lookup/12915*'
_output_shapes
:��������� �
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup=^sequential/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2|
<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2<sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12806
queries
sequential_12788
sequential_12790	"
sequential_12792:e 0
matmul_readvariableop_resource: 
gather_resource:

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_12788sequential_12790sequential_12792*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12662t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12744
input_1
sequential_12726
sequential_12728	"
sequential_12730:e 0
matmul_readvariableop_resource: 
gather_resource:

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12726sequential_12728sequential_12730*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12662t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
!__inference__traced_restore_13102
file_prefix*
assignvariableop_identifiers:/
assignvariableop_1_candidates: 9
'assignvariableop_2_embedding_embeddings:e 

identity_4��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp'assignvariableop_2_embedding_embeddingsIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_4Identity_4:output:0*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_13018

inputs	(
embedding_lookup_13012:e 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_13012inputs*
Tindices0	*)
_class
loc:@embedding_lookup/13012*'
_output_shapes
:��������� *
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/13012*'
_output_shapes
:��������� }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:��������� Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_12640

inputsB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	!
embedding_12636:e 
identity��!embedding/StatefulPartitionedCall�1string_lookup/hash_table_Lookup/LookupTableFindV2�
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinputs?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_12636*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12610y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
__inference__initializer_130316
2key_value_init115_lookuptableimportv2_table_handle.
*key_value_init115_lookuptableimportv2_keys0
,key_value_init115_lookuptableimportv2_values	
identity��%key_value_init115/LookupTableImportV2�
%key_value_init115/LookupTableImportV2LookupTableImportV22key_value_init115_lookuptableimportv2_table_handle*key_value_init115_lookuptableimportv2_keys,key_value_init115_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init115/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :d:d2N
%key_value_init115/LookupTableImportV2%key_value_init115/LookupTableImportV2: 

_output_shapes
:d: 

_output_shapes
:d
�
�
*__inference_sequential_layer_call_fn_12671
string_lookup_input
unknown
	unknown_0	
	unknown_1:e 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12662o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_12626
string_lookup_inputB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	!
embedding_12622:e 
identity��!embedding/StatefulPartitionedCall�1string_lookup/hash_table_Lookup/LookupTableFindV2�
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handlestring_lookup_input?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_12622*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12610y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12723
input_1
sequential_12705
sequential_12707	"
sequential_12709:e 0
matmul_readvariableop_resource: 
gather_resource:

identity_1

identity_2��Gather�MatMul/ReadVariableOp�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12705sequential_12707sequential_12709*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12640t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:���������
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
*__inference_sequential_layer_call_fn_12965

inputs
unknown
	unknown_0	
	unknown_1:e 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12640o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�

�
+__inference_brute_force_layer_call_fn_12908
queries
unknown
	unknown_0	
	unknown_1:e 
	unknown_2: 
	unknown_3:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_brute_force_layer_call_and_return_conditional_losses_12806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�

�
+__inference_brute_force_layer_call_fn_12891
queries
unknown
	unknown_0	
	unknown_1:e 
	unknown_2: 
	unknown_3:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_brute_force_layer_call_and_return_conditional_losses_12768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	queries:

_output_shapes
: 
�

�
+__inference_brute_force_layer_call_fn_12783
input_1
unknown
	unknown_0	
	unknown_1:e 
	unknown_2: 
	unknown_3:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_brute_force_layer_call_and_return_conditional_losses_12768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_12615
string_lookup_inputB
>string_lookup_hash_table_lookup_lookuptablefindv2_table_handleC
?string_lookup_hash_table_lookup_lookuptablefindv2_default_value	!
embedding_12611:e 
identity��!embedding/StatefulPartitionedCall�1string_lookup/hash_table_Lookup/LookupTableFindV2�
1string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2>string_lookup_hash_table_lookup_lookuptablefindv2_table_handlestring_lookup_input?string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
string_lookup/IdentityIdentity:string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_12611*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12610y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp"^embedding/StatefulPartitionedCall2^string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2f
1string_lookup/hash_table_Lookup/LookupTableFindV21string_lookup/hash_table_Lookup/LookupTableFindV2:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�	
�
#__inference_signature_wrapper_12874
input_1
unknown
	unknown_0	
	unknown_1:e 
	unknown_2: 
	unknown_3:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_12593o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�
�
*__inference_sequential_layer_call_fn_12649
string_lookup_input
unknown
	unknown_0	
	unknown_1:e 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12640o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:���������
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
�
:
__inference__creator_13023
identity��
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name116*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
*__inference_sequential_layer_call_fn_12976

inputs
unknown
	unknown_0	
	unknown_1:e 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12662o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
 __inference__wrapped_model_12593
input_1Y
Ubrute_force_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleZ
Vbrute_force_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	I
7brute_force_sequential_embedding_embedding_lookup_12577:e <
*brute_force_matmul_readvariableop_resource: )
brute_force_gather_resource:
identity

identity_1��brute_force/Gather�!brute_force/MatMul/ReadVariableOp�1brute_force/sequential/embedding/embedding_lookup�Hbrute_force/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2�
Hbrute_force/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Ubrute_force_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleinput_1Vbrute_force_sequential_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
-brute_force/sequential/string_lookup/IdentityIdentityQbrute_force/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
1brute_force/sequential/embedding/embedding_lookupResourceGather7brute_force_sequential_embedding_embedding_lookup_125776brute_force/sequential/string_lookup/Identity:output:0*
Tindices0	*J
_class@
><loc:@brute_force/sequential/embedding/embedding_lookup/12577*'
_output_shapes
:��������� *
dtype0�
:brute_force/sequential/embedding/embedding_lookup/IdentityIdentity:brute_force/sequential/embedding/embedding_lookup:output:0*
T0*J
_class@
><loc:@brute_force/sequential/embedding/embedding_lookup/12577*'
_output_shapes
:��������� �
<brute_force/sequential/embedding/embedding_lookup/Identity_1IdentityCbrute_force/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� �
!brute_force/MatMul/ReadVariableOpReadVariableOp*brute_force_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
brute_force/MatMulMatMulEbrute_force/sequential/embedding/embedding_lookup/Identity_1:output:0)brute_force/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������*
transpose_b(V
brute_force/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
�
brute_force/TopKV2TopKV2brute_force/MatMul:product:0brute_force/TopKV2/k:output:0*
T0*:
_output_shapes(
&:���������
:���������
�
brute_force/GatherResourceGatherbrute_force_gather_resourcebrute_force/TopKV2:indices:0*
Tindices0*'
_output_shapes
:���������
*
dtype0o
brute_force/IdentityIdentitybrute_force/Gather:output:0*
T0*'
_output_shapes
:���������
j
IdentityIdentitybrute_force/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:���������
n

Identity_1Identitybrute_force/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^brute_force/Gather"^brute_force/MatMul/ReadVariableOp2^brute_force/sequential/embedding/embedding_lookupI^brute_force/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 2(
brute_force/Gatherbrute_force/Gather2F
!brute_force/MatMul/ReadVariableOp!brute_force/MatMul/ReadVariableOp2f
1brute_force/sequential/embedding/embedding_lookup1brute_force/sequential/embedding/embedding_lookup2�
Hbrute_force/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2Hbrute_force/sequential/string_lookup/hash_table_Lookup/LookupTableFindV2:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�

�
+__inference_brute_force_layer_call_fn_12821
input_1
unknown
	unknown_0	
	unknown_1:e 
	unknown_2: 
	unknown_3:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������
:���������
*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_brute_force_layer_call_and_return_conditional_losses_12806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:

_output_shapes
: 
�%
�
__inference__traced_save_13083
file_prefix0
"read_disablecopyonread_identifiers:5
#read_1_disablecopyonread_candidates: ?
-read_2_disablecopyonread_embedding_embeddings:e 
savev2_const_3

identity_7��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: t
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_identifiers"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_identifiers^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_candidates"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_candidates^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_2/DisableCopyOnReadDisableCopyOnRead-read_2_disablecopyonread_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp-read_2_disablecopyonread_embedding_embeddings^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:e *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:e �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0savev2_const_3"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_6Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_7IdentityIdentity_6:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*
_input_shapes

: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
�
,
__inference__destroyer_13036
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
}
)__inference_embedding_layer_call_fn_13009

inputs	
unknown:e 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
input_1,
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������
<
output_20
StatefulPartitionedCall:1���������
tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
query_model
	identifiers
	_identifiers


candidates

_candidates
query_with_exclusions

signatures"
_tf_keras_model
5
0
	1

2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
+__inference_brute_force_layer_call_fn_12783
+__inference_brute_force_layer_call_fn_12821
+__inference_brute_force_layer_call_fn_12891
+__inference_brute_force_layer_call_fn_12908�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
F__inference_brute_force_layer_call_and_return_conditional_losses_12723
F__inference_brute_force_layer_call_and_return_conditional_losses_12744
F__inference_brute_force_layer_call_and_return_conditional_losses_12931
F__inference_brute_force_layer_call_and_return_conditional_losses_12954�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
	capture_1B�
 __inference__wrapped_model_12593input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_sequential
:2identifiers
: 2
candidates
�2��
���
FullArgSpec)
args!�
	jqueries
j
exclusions
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
$serving_default"
signature_map
&:$e 2embedding/embeddings
.
	0

1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
+__inference_brute_force_layer_call_fn_12783input_1"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
+__inference_brute_force_layer_call_fn_12821input_1"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
+__inference_brute_force_layer_call_fn_12891queries"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
+__inference_brute_force_layer_call_fn_12908queries"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
F__inference_brute_force_layer_call_and_return_conditional_losses_12723input_1"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
F__inference_brute_force_layer_call_and_return_conditional_losses_12744input_1"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
F__inference_brute_force_layer_call_and_return_conditional_losses_12931queries"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
�
	capture_1B�
F__inference_brute_force_layer_call_and_return_conditional_losses_12954queries"�
���
FullArgSpec
args�
	jqueries
jk
varargs
 
varkw
 
defaults�

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_1
!J	
Const_2jtf.TrackableConstant
:
%	keras_api
&lookup_table"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
2trace_0
3trace_1
4trace_2
5trace_32�
*__inference_sequential_layer_call_fn_12649
*__inference_sequential_layer_call_fn_12671
*__inference_sequential_layer_call_fn_12965
*__inference_sequential_layer_call_fn_12976�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z2trace_0z3trace_1z4trace_2z5trace_3
�
6trace_0
7trace_1
8trace_2
9trace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_12615
E__inference_sequential_layer_call_and_return_conditional_losses_12626
E__inference_sequential_layer_call_and_return_conditional_losses_12989
E__inference_sequential_layer_call_and_return_conditional_losses_13002�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z6trace_0z7trace_1z8trace_2z9trace_3
�
	capture_1B�
#__inference_signature_wrapper_12874input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
"
_generic_user_object
f
:_initializer
;_create_resource
<_initialize
=_destroy_resourceR jtf.StaticHashTable
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
)__inference_embedding_layer_call_fn_13009�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0
�
Dtrace_02�
D__inference_embedding_layer_call_and_return_conditional_losses_13018�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_1B�
*__inference_sequential_layer_call_fn_12649string_lookup_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
*__inference_sequential_layer_call_fn_12671string_lookup_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
*__inference_sequential_layer_call_fn_12965inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
*__inference_sequential_layer_call_fn_12976inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_12615string_lookup_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_12626string_lookup_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_12989inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
�
	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_13002inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_1
"
_generic_user_object
�
Etrace_02�
__inference__creator_13023�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zEtrace_0
�
Ftrace_02�
__inference__initializer_13031�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zFtrace_0
�
Gtrace_02�
__inference__destroyer_13036�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zGtrace_0
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
�B�
)__inference_embedding_layer_call_fn_13009inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_embedding_layer_call_and_return_conditional_losses_13018inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference__creator_13023"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
H	capture_1
I	capture_2B�
__inference__initializer_13031"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zH	capture_1zI	capture_2
�B�
__inference__destroyer_13036"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant?
__inference__creator_13023!�

� 
� "�
unknown A
__inference__destroyer_13036!�

� 
� "�
unknown H
__inference__initializer_13031&&HI�

� 
� "�
unknown �
 __inference__wrapped_model_12593�&
	,�)
"�
�
input_1���������
� "c�`
.
output_1"�
output_1���������

.
output_2"�
output_2���������
�
F__inference_brute_force_layer_call_and_return_conditional_losses_12723�&
	@�=
&�#
�
input_1���������

 
�

trainingp"Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
F__inference_brute_force_layer_call_and_return_conditional_losses_12744�&
	@�=
&�#
�
input_1���������

 
�

trainingp "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
F__inference_brute_force_layer_call_and_return_conditional_losses_12931�&
	@�=
&�#
�
queries���������

 
�

trainingp"Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
F__inference_brute_force_layer_call_and_return_conditional_losses_12954�&
	@�=
&�#
�
queries���������

 
�

trainingp "Y�V
O�L
$�!

tensor_0_0���������

$�!

tensor_0_1���������

� �
+__inference_brute_force_layer_call_fn_12783�&
	@�=
&�#
�
input_1���������

 
�

trainingp"K�H
"�
tensor_0���������

"�
tensor_1���������
�
+__inference_brute_force_layer_call_fn_12821�&
	@�=
&�#
�
input_1���������

 
�

trainingp "K�H
"�
tensor_0���������

"�
tensor_1���������
�
+__inference_brute_force_layer_call_fn_12891�&
	@�=
&�#
�
queries���������

 
�

trainingp"K�H
"�
tensor_0���������

"�
tensor_1���������
�
+__inference_brute_force_layer_call_fn_12908�&
	@�=
&�#
�
queries���������

 
�

trainingp "K�H
"�
tensor_0���������

"�
tensor_1���������
�
D__inference_embedding_layer_call_and_return_conditional_losses_13018^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0��������� 
� �
)__inference_embedding_layer_call_fn_13009S+�(
!�
�
inputs���������	
� "!�
unknown��������� �
E__inference_sequential_layer_call_and_return_conditional_losses_12615u&@�=
6�3
)�&
string_lookup_input���������
p

 
� ",�)
"�
tensor_0��������� 
� �
E__inference_sequential_layer_call_and_return_conditional_losses_12626u&@�=
6�3
)�&
string_lookup_input���������
p 

 
� ",�)
"�
tensor_0��������� 
� �
E__inference_sequential_layer_call_and_return_conditional_losses_12989h&3�0
)�&
�
inputs���������
p

 
� ",�)
"�
tensor_0��������� 
� �
E__inference_sequential_layer_call_and_return_conditional_losses_13002h&3�0
)�&
�
inputs���������
p 

 
� ",�)
"�
tensor_0��������� 
� �
*__inference_sequential_layer_call_fn_12649j&@�=
6�3
)�&
string_lookup_input���������
p

 
� "!�
unknown��������� �
*__inference_sequential_layer_call_fn_12671j&@�=
6�3
)�&
string_lookup_input���������
p 

 
� "!�
unknown��������� �
*__inference_sequential_layer_call_fn_12965]&3�0
)�&
�
inputs���������
p

 
� "!�
unknown��������� �
*__inference_sequential_layer_call_fn_12976]&3�0
)�&
�
inputs���������
p 

 
� "!�
unknown��������� �
#__inference_signature_wrapper_12874�&
	7�4
� 
-�*
(
input_1�
input_1���������"c�`
.
output_1"�
output_1���������

.
output_2"�
output_2���������
